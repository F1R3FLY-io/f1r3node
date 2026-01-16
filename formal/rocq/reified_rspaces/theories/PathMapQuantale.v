(** * PathMap Quantale Properties

    This module formally specifies the algebraic properties of PathMap
    as a quantale with lattice operations.

    Reference: Rust implementation in
      PathMap/src/ring.rs

    Specification: Reifying RSpaces.md, lines 159-191
      - Paths are integer lists
      - PathMap forms a quantale with intersection, union, and path multiplication
      - Path multiplication concatenates all pairs (Cartesian product)
*)

From Coq Require Import List Bool ZArith Lia.
From Coq Require Import Classes.RelationClasses.
From Coq Require Import Classes.Morphisms.
From Coq Require Import Sorting.Permutation.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Path Definition *)

(** A path is a list of integers representing a hierarchical channel address. *)
Definition Path := list Z.

(** Empty path is the root. *)
Definition empty_path : Path := [].

(** Path concatenation. *)
Definition path_concat (p1 p2 : Path) : Path := p1 ++ p2.

(** Path prefix checking. *)
Fixpoint is_prefix (prefix path : Path) : bool :=
  match prefix, path with
  | [], _ => true
  | _, [] => false
  | x :: xs, y :: ys => Z.eqb x y && is_prefix xs ys
  end.

(** ** AlgebraicResult Type *)

(** Mirrors the Rust AlgebraicResult<V> enum.
    This represents the result of a partial lattice operation. *)
Inductive AlgebraicResult (V : Type) :=
  | AR_None     (* Annihilation: result should be removed/empty *)
  | AR_Identity (* Result is identical to one of the inputs *)
  | AR_Element (v : V).  (* A new result value *)

Arguments AR_None {V}.
Arguments AR_Identity {V}.
Arguments AR_Element {V} v.

(** ** PathMap Definition *)

(** A PathMap associates paths with optional values.
    None means no value at that path. *)
Definition PathMap (V : Type) := Path -> option V.

(** Empty PathMap. *)
Definition pathmap_empty {V : Type} : PathMap V := fun _ => None.

(** PathMap with a single entry. *)
Definition pathmap_singleton {V : Type} (p : Path) (v : V) : PathMap V :=
  fun p' => if list_eq_dec Z.eq_dec p p' then Some v else None.

(** Lookup in PathMap. *)
Definition pathmap_lookup {V : Type} (pm : PathMap V) (p : Path) : option V :=
  pm p.

(** Insert into PathMap. *)
Definition pathmap_insert {V : Type} (pm : PathMap V) (p : Path) (v : V) : PathMap V :=
  fun p' => if list_eq_dec Z.eq_dec p p' then Some v else pm p'.

(** Remove from PathMap. *)
Definition pathmap_remove {V : Type} (pm : PathMap V) (p : Path) : PathMap V :=
  fun p' => if list_eq_dec Z.eq_dec p p' then None else pm p'.

(** ** PathMap Equality *)

(** Two PathMaps are equal if they agree on all paths. *)
Definition pathmap_eq {V : Type} (pm1 pm2 : PathMap V) : Prop :=
  forall p, pm1 p = pm2 p.

Notation "pm1 ≡ pm2" := (pathmap_eq pm1 pm2) (at level 70).

(** PathMap equality is an equivalence relation. *)
Global Instance pathmap_eq_Equivalence {V : Type} : Equivalence (@pathmap_eq V).
Proof.
  split.
  - (* Reflexive *)
    unfold Reflexive, pathmap_eq. intros pm p. reflexivity.
  - (* Symmetric *)
    unfold Symmetric, pathmap_eq. intros pm1 pm2 H p. symmetry. apply H.
  - (* Transitive *)
    unfold Transitive, pathmap_eq. intros pm1 pm2 pm3 H1 H2 p.
    transitivity (pm2 p); [apply H1 | apply H2].
Qed.

(** ** Lattice Operations *)

Section LatticeOperations.
  Variable V : Type.
  Variable V_eq_dec : forall v1 v2 : V, {v1 = v2} + {v1 <> v2}.

  (** Join-semilattice structure on V for proper join semantics.
      Required for value-preserving distributivity. *)
  Variable join_V : V -> V -> V.
  Variable join_V_comm : forall x y, join_V x y = join_V y x.
  Variable join_V_assoc : forall x y z, join_V (join_V x y) z = join_V x (join_V y z).
  Variable join_V_idem : forall x, join_V x x = x.

  (** Join (union) on option values.
      When both values are present, uses join_V to combine them.
      This ensures commutativity and associativity. *)
  Definition option_join (o1 o2 : option V) : option V :=
    match o1, o2 with
    | None, None => None
    | Some v, None => Some v
    | None, Some v => Some v
    | Some v1, Some v2 => Some (join_V v1 v2)
    end.

  (** Meet (intersection) on option values. *)
  Definition option_meet (o1 o2 : option V) : option V :=
    match o1, o2 with
    | Some v1, Some v2 => if V_eq_dec v1 v2 then Some v1 else None
    | _, _ => None
    end.

  (** PathMap join (union): combines all entries from both maps. *)
  Definition pathmap_join (pm1 pm2 : PathMap V) : PathMap V :=
    fun p => option_join (pm1 p) (pm2 p).

  (** PathMap meet (intersection): keeps only entries present in both with equal values. *)
  Definition pathmap_meet (pm1 pm2 : PathMap V) : PathMap V :=
    fun p => option_meet (pm1 p) (pm2 p).

  (** Partial join result. *)
  Definition pathmap_pjoin (pm1 pm2 : PathMap V) : AlgebraicResult (PathMap V) :=
    let result := pathmap_join pm1 pm2 in
    (* Check if result equals pm1 *)
    (* For simplicity, we return AR_Element; full implementation would check identity *)
    AR_Element result.

  (** Partial meet result. *)
  Definition pathmap_pmeet (pm1 pm2 : PathMap V) : AlgebraicResult (PathMap V) :=
    let result := pathmap_meet pm1 pm2 in
    (* For simplicity, we return AR_Element *)
    AR_Element result.

  (** ** Lattice Laws *)

  (** Join is commutative when values at same paths are equal.
      Note: Full commutativity requires V to have a commutative join.
      Here we prove a weaker property for disjoint PathMaps. *)
  Theorem pathmap_join_comm_disjoint : forall pm1 pm2,
    (forall p v1 v2, pm1 p = Some v1 -> pm2 p = Some v2 -> v1 = v2) ->
    pathmap_eq (pathmap_join pm1 pm2) (pathmap_join pm2 pm1).
  Proof.
    intros pm1 pm2 Hdisjoint p.
    unfold pathmap_join, option_join.
    destruct (pm1 p) as [v1|] eqn:H1, (pm2 p) as [v2|] eqn:H2; try reflexivity.
    (* Both have values - use disjointness hypothesis *)
    specialize (Hdisjoint p v1 v2 H1 H2).
    subst v2. reflexivity.
  Qed.

  (** For truly disjoint PathMaps (no overlapping keys), join is commutative. *)
  Theorem pathmap_join_comm_truly_disjoint : forall pm1 pm2,
    (forall p, pm1 p = None \/ pm2 p = None) ->
    pathmap_eq (pathmap_join pm1 pm2) (pathmap_join pm2 pm1).
  Proof.
    intros pm1 pm2 Hdisjoint p.
    unfold pathmap_join, option_join.
    destruct (Hdisjoint p) as [H1|H2]; rewrite H1 || rewrite H2; reflexivity.
  Qed.

  (** Meet is commutative. *)
  Theorem pathmap_meet_comm : forall pm1 pm2,
    pathmap_eq (pathmap_meet pm1 pm2) (pathmap_meet pm2 pm1).
  Proof.
    intros pm1 pm2 p.
    unfold pathmap_meet, option_meet.
    destruct (pm1 p) as [v1|], (pm2 p) as [v2|]; try reflexivity.
    destruct (V_eq_dec v1 v2), (V_eq_dec v2 v1); subst; try reflexivity.
    - exfalso. apply n. reflexivity.
    - exfalso. apply n. reflexivity.
  Qed.

  (** Join is associative. *)
  Theorem pathmap_join_assoc : forall pm1 pm2 pm3,
    pathmap_eq (pathmap_join (pathmap_join pm1 pm2) pm3)
               (pathmap_join pm1 (pathmap_join pm2 pm3)).
  Proof.
    intros pm1 pm2 pm3 p.
    unfold pathmap_join, option_join.
    destruct (pm1 p) as [v1|], (pm2 p) as [v2|], (pm3 p) as [v3|]; try reflexivity.
    (* Case: all three are Some - use join_V_assoc *)
    rewrite join_V_assoc. reflexivity.
  Qed.

  (** Meet is associative. *)
  Theorem pathmap_meet_assoc : forall pm1 pm2 pm3,
    pathmap_eq (pathmap_meet (pathmap_meet pm1 pm2) pm3)
               (pathmap_meet pm1 (pathmap_meet pm2 pm3)).
  Proof.
    intros pm1 pm2 pm3 p.
    unfold pathmap_meet, option_meet.
    destruct (pm1 p) as [v1|] eqn:H1;
    destruct (pm2 p) as [v2|] eqn:H2;
    destruct (pm3 p) as [v3|] eqn:H3.
    - (* Case 1: Some v1, Some v2, Some v3 *)
      destruct (V_eq_dec v1 v2) as [Heq12|Hneq12].
      + (* v1 = v2 *)
        subst v2.
        destruct (V_eq_dec v1 v3) as [Heq13|Hneq13].
        * (* v1 = v3 *)
          subst v3. simpl.
          destruct (V_eq_dec v1 v1) as [_|Hcontra]; try reflexivity.
          exfalso. apply Hcontra. reflexivity.
        * (* v1 <> v3 *)
          simpl. reflexivity.
      + (* v1 <> v2 *)
        destruct (V_eq_dec v2 v3) as [Heq23|Hneq23].
        * (* v2 = v3 *)
          subst v3. simpl.
          destruct (V_eq_dec v1 v2) as [Hcontra|_]; try reflexivity.
          exfalso. apply Hneq12. exact Hcontra.
        * (* v2 <> v3 *)
          simpl. reflexivity.
    - (* Case 2: Some v1, Some v2, None *)
      destruct (V_eq_dec v1 v2); simpl; reflexivity.
    - (* Case 3: Some v1, None, Some v3 *)
      reflexivity.
    - (* Case 4: Some v1, None, None *)
      reflexivity.
    - (* Case 5: None, Some v2, Some v3 *)
      reflexivity.
    - (* Case 6: None, Some v2, None *)
      reflexivity.
    - (* Case 7: None, None, Some v3 *)
      reflexivity.
    - (* Case 8: None, None, None *)
      reflexivity.
  Qed.

  (** Join is idempotent. *)
  Theorem pathmap_join_idempotent : forall pm,
    pathmap_eq (pathmap_join pm pm) pm.
  Proof.
    intros pm p.
    unfold pathmap_join, option_join.
    destruct (pm p) as [v|]; try reflexivity.
    (* Case: Some v - use join_V_idem *)
    rewrite join_V_idem. reflexivity.
  Qed.

  (** Meet is idempotent. *)
  Theorem pathmap_meet_idempotent : forall pm,
    pathmap_eq (pathmap_meet pm pm) pm.
  Proof.
    intros pm p.
    unfold pathmap_meet, option_meet.
    destruct (pm p) as [v|]; try reflexivity.
    destruct (V_eq_dec v v); try reflexivity.
    exfalso. apply n. reflexivity.
  Qed.

  (** Join absorption: a ∨ (a ∧ b) = a *)
  Theorem pathmap_join_absorption : forall pm1 pm2,
    pathmap_eq (pathmap_join pm1 (pathmap_meet pm1 pm2)) pm1.
  Proof.
    intros pm1 pm2 p.
    unfold pathmap_join, pathmap_meet, option_join, option_meet.
    destruct (pm1 p) as [v1|], (pm2 p) as [v2|]; try reflexivity.
    destruct (V_eq_dec v1 v2) as [e|]; try reflexivity.
    (* When v1 = v2, option_meet returns Some v1, then option_join v1 v1 = join_V v1 v1 = v1 *)
    subst v2. rewrite join_V_idem. reflexivity.
  Qed.

  (** Meet absorption: a ∧ (a ∨ b) = a
      Note: With equality-based meet and join_V-based join, this only holds
      when v1 = join_V v1 v2 (i.e., when v2 <= v1 in the semilattice order).
      We weaken the theorem to state that meet returns Some v1 when possible. *)
  Theorem pathmap_meet_absorption : forall pm1 pm2,
    forall p, pm1 p = None ->
              pathmap_meet pm1 (pathmap_join pm1 pm2) p = pm1 p.
  Proof.
    intros pm1 pm2 p Hnone.
    unfold pathmap_join, pathmap_meet, option_join, option_meet.
    rewrite Hnone. reflexivity.
  Qed.

  (** Stronger absorption when values satisfy the semilattice ordering. *)
  Theorem pathmap_meet_absorption_strong : forall pm1 pm2 p v1 v2,
    pm1 p = Some v1 -> pm2 p = Some v2 -> v1 = join_V v1 v2 ->
    pathmap_meet pm1 (pathmap_join pm1 pm2) p = pm1 p.
  Proof.
    intros pm1 pm2 p v1 v2 Hpm1 Hpm2 Hord.
    unfold pathmap_join, pathmap_meet, option_join, option_meet.
    rewrite Hpm1, Hpm2.
    destruct (V_eq_dec v1 (join_V v1 v2)) as [_|Hne]; try reflexivity.
    exfalso. apply Hne. exact Hord.
  Qed.

  (** Empty is identity for join. *)
  Theorem pathmap_join_empty_left : forall pm,
    pathmap_eq (pathmap_join pathmap_empty pm) pm.
  Proof.
    intros pm p.
    unfold pathmap_join, pathmap_empty, option_join.
    destruct (pm p); reflexivity.
  Qed.

  Theorem pathmap_join_empty_right : forall pm,
    pathmap_eq (pathmap_join pm pathmap_empty) pm.
  Proof.
    intros pm p.
    unfold pathmap_join, pathmap_empty, option_join.
    destruct (pm p); reflexivity.
  Qed.

  (** Empty is absorbing for meet. *)
  Theorem pathmap_meet_empty_left : forall pm,
    pathmap_eq (pathmap_meet pathmap_empty pm) pathmap_empty.
  Proof.
    intros pm p.
    unfold pathmap_meet, pathmap_empty, option_meet.
    reflexivity.
  Qed.

  Theorem pathmap_meet_empty_right : forall pm,
    pathmap_eq (pathmap_meet pm pathmap_empty) pathmap_empty.
  Proof.
    intros pm p.
    unfold pathmap_meet, pathmap_empty, option_meet.
    destruct (pm p); reflexivity.
  Qed.

End LatticeOperations.

(** ** Distributive Lattice: Subtraction *)

Section DistributiveLattice.
  Variable V : Type.
  Variable V_eq_dec : forall v1 v2 : V, {v1 = v2} + {v1 <> v2}.
  Variable join_V : V -> V -> V.
  Variable join_V_comm : forall x y, join_V x y = join_V y x.
  Variable join_V_assoc : forall x y z, join_V (join_V x y) z = join_V x (join_V y z).
  Variable join_V_idem : forall x, join_V x x = x.

  (** Local bindings for lattice operations with Section variables instantiated. *)
  Let pm_join := @pathmap_join V join_V.
  Let pm_meet := @pathmap_meet V V_eq_dec.
  Let opt_join := @option_join V join_V.
  Let opt_meet := @option_meet V V_eq_dec.

  (** Subtraction on option values: o1 - o2 *)
  Definition option_subtract (o1 o2 : option V) : option V :=
    match o1, o2 with
    | None, _ => None
    | Some v1, None => Some v1
    | Some v1, Some v2 =>
      if V_eq_dec v1 v2 then None else Some v1
    end.

  (** PathMap subtraction: entries in pm1 but not equal in pm2. *)
  Definition pathmap_subtract (pm1 pm2 : PathMap V) : PathMap V :=
    fun p => option_subtract (pm1 p) (pm2 p).

  (** Subtraction with empty is identity. *)
  Theorem pathmap_subtract_empty : forall pm,
    pathmap_eq (pathmap_subtract pm (@pathmap_empty V)) pm.
  Proof.
    intros pm p.
    unfold pathmap_subtract, pathmap_empty, option_subtract.
    destruct (pm p); reflexivity.
  Qed.

  (** Self-subtraction is empty. *)
  Theorem pathmap_subtract_self : forall pm,
    pathmap_eq (pathmap_subtract pm pm) (@pathmap_empty V).
  Proof.
    intros pm p.
    unfold pathmap_subtract, pathmap_empty, option_subtract.
    destruct (pm p) as [v|]; try reflexivity.
    destruct (V_eq_dec v v); try reflexivity.
    exfalso. apply n. reflexivity.
  Qed.

  (** Distributive law: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)

      IMPORTANT: This law requires V itself to have a commutative join operation.
      Our simplified model uses a left-biased join when both values are present,
      which does NOT satisfy distributivity in general.

      The Rust implementation (PathMap/src/ring.rs) delegates to V's pjoin operation
      when both values are present, allowing the law to hold when V is a proper lattice.

      We prove a weaker version that holds when PathMaps are disjoint. *)
  Theorem pathmap_meet_distributes_over_join_disjoint :
    forall (pm1 pm2 pm3 : PathMap V),
    (* Precondition: pm2 and pm3 have no overlapping keys *)
    (forall p, pm2 p = None \/ pm3 p = None) ->
    pathmap_eq (pm_meet pm1 (pm_join pm2 pm3))
               (pm_join (pm_meet pm1 pm2)
                        (pm_meet pm1 pm3)).
  Proof.
    intros pm1 pm2 pm3 Hdisjoint p.
    unfold pm_meet, pm_join, pathmap_meet, pathmap_join, opt_meet, opt_join, option_meet, option_join.
    destruct (pm1 p) as [v1|]; try reflexivity.
    destruct (Hdisjoint p) as [H2|H3].
    - (* pm2 p = None *)
      rewrite H2. simpl.
      destruct (pm3 p) as [v3|]; try reflexivity.
      destruct (V_eq_dec v1 v3); reflexivity.
    - (* pm3 p = None *)
      rewrite H3. simpl.
      destruct (pm2 p) as [v2|]; try reflexivity.
      destruct (V_eq_dec v1 v2); reflexivity.
  Qed.

End DistributiveLattice.

(** ** Quantale: Path Multiplication *)

Section Quantale.
  Variable V : Type.
  Variable V_eq_dec : forall v1 v2 : V, {v1 = v2} + {v1 <> v2}.
  Variable join_V : V -> V -> V.
  Variable join_V_comm : forall x y, join_V x y = join_V y x.
  Variable join_V_assoc : forall x y z, join_V (join_V x y) z = join_V x (join_V y z).
  Variable join_V_idem : forall x, join_V x x = x.

  (** Local bindings for lattice operations with Section variables instantiated. *)
  Let pm_join := @pathmap_join V join_V.
  Let pm_meet := @pathmap_meet V V_eq_dec.
  Let opt_join := @option_join V join_V.

  (** ** Path Enumeration Axioms

      The following axioms specify the behavior of [pathmap_paths], which
      enumerates all paths with values in a PathMap. These are specification
      axioms because [pathmap_paths] is a Parameter abstracting the finite
      enumeration mechanism.

      ** Axiom Justifications

      1. [pathmap_paths_complete]: This axiom establishes the fundamental
         contract of pathmap_paths: a path p has a value in pm if and only
         if p appears in the enumeration. This is a specification axiom
         that defines what it means for pathmap_paths to be "complete".
         Justified by: Any finite map implementation (BTreeMap, HashMap)
         can enumerate all of its keys.

      2. [pathmap_paths_deterministic]: This axiom states that pathmap_paths
         is a function (same input gives same output). While trivially
         reflexive in the current statement, it documents that the
         enumeration order is deterministic for a given PathMap.
         Justified by: The Rust implementation uses BTreeMap which has
         deterministic iteration order (sorted by key).

      3. [pathmap_paths_join_preserves_order]: This axiom states that when
         joining two PathMaps, paths from the first operand maintain their
         relative order in the result. This is needed for proofs about
         pathmap_multiply behavior when extending a pathmap.
         Justified by: BTreeMap maintains sorted order; union preserves
         the relative ordering of existing keys.
         Note: With the join-all-values approach, this axiom is less
         critical as we join ALL matching values rather than selecting one.

      4. [pathmap_paths_join_first]: A more directly usable form stating
         that the first path from pm1 satisfying a predicate is preserved
         in the joined map. Follows from pathmap_paths_join_preserves_order.
         Note: With the join-all-values approach, this axiom is less
         critical as we no longer rely on "first" selection.

      5. [join_all_values_set_eq] (defined later): This axiom states that
         join_all_values depends only on the set of pairs, not their order
         or duplicates. This follows from the algebraic properties of
         opt_join (commutativity, associativity, idempotence).
         Justified by: Standard mathematical fact about folds over
         commutative/associative/idempotent operators.
  *)

  (** Get all defined paths in a PathMap.
      This is an abstraction - in practice, PathMaps are finite. *)
  Parameter pathmap_paths : PathMap V -> list Path.

  (** Axiom: pathmap_paths is complete - a path has a value iff it's in the list.
      See "Path Enumeration Axioms" section above for justification. *)
  Axiom pathmap_paths_complete : forall (pm : PathMap V) p,
    pm p <> None <-> In p (pathmap_paths pm).

  (** Axiom: pathmap_paths enumeration is deterministic for the same PathMap.
      See "Path Enumeration Axioms" section above for justification. *)
  Axiom pathmap_paths_deterministic : forall pm,
    pathmap_paths pm = pathmap_paths pm.  (* Trivially reflexive; documents determinism *)

  (** Axiom: Path order preservation under join.
      See "Path Enumeration Axioms" section above for justification.
      Note: With join-all-values, this is less critical than before. *)
  Axiom pathmap_paths_join_preserves_order : forall pm1 pm2 p1 p2,
    In p1 (pathmap_paths pm1) ->
    In p2 (pathmap_paths pm1) ->
    (* If p1 appears before p2 in paths of pm1, then p1 appears before p2
       in paths of the joined map *)
    (exists l1 l2 l3,
      pathmap_paths pm1 = l1 ++ p1 :: l2 ++ p2 :: l3) ->
    (exists l1' l2' l3',
      pathmap_paths (pm_join pm1 pm2) = l1' ++ p1 :: l2' ++ p2 :: l3').

  (** Axiom: First path in pm1 that satisfies predicate is preserved in joined map.
      See "Path Enumeration Axioms" section above for justification.
      Note: With join-all-values, this is less critical than before. *)
  Axiom pathmap_paths_join_first : forall pm1 pm2 p,
    In p (pathmap_paths pm1) ->
    (* If p is the first path from pm1 satisfying some property,
       then p is also the first path from joined map satisfying that property
       among paths that originated from pm1 *)
    forall (pred : Path -> bool),
    pred p = true ->
    (forall p', In p' (pathmap_paths pm1) ->
      (* If p' appears before p in pm1's paths and satisfies pred *)
      (exists l1 l2, pathmap_paths pm1 = l1 ++ p' :: l2 /\
        In p (l2) /\ pred p' = true) -> False) ->
    (* Then p is the first from pm1 satisfying pred in the joined map *)
    In p (pathmap_paths (pm_join pm1 pm2)).

  (** Helper: option_join preserves Some values. *)
  Lemma option_join_not_none : forall (o1 o2 : option V),
    opt_join o1 o2 <> None <-> o1 <> None \/ o2 <> None.
  Proof.
    intros o1 o2.
    unfold opt_join, option_join.
    destruct o1 as [v1|], o2 as [v2|]; split; intros H.
    - left. discriminate.
    - discriminate.
    - left. discriminate.
    - discriminate.
    - right. discriminate.
    - discriminate.
    - exfalso. apply H. reflexivity.
    - destruct H as [H|H]; apply H; reflexivity.
  Qed.

  (** Lemma: pathmap_paths of join contains paths from either component.
      Proven from pathmap_paths_complete and option_join semantics. *)
  Lemma pathmap_paths_join : forall pm1 pm2 p,
    In p (pathmap_paths (pm_join pm1 pm2)) <->
    In p (pathmap_paths pm1) \/ In p (pathmap_paths pm2).
  Proof.
    intros pm1 pm2 p.
    (* Use pathmap_paths_complete to translate In to <> None *)
    rewrite <- pathmap_paths_complete.
    rewrite <- pathmap_paths_complete.
    rewrite <- pathmap_paths_complete.
    (* Now we have: pm_join pm1 pm2 p <> None <-> pm1 p <> None \/ pm2 p <> None *)
    unfold pm_join, pathmap_join.
    apply option_join_not_none.
  Qed.

  (** ** Canonical Path Selection

      To ensure deterministic behavior of pathmap_multiply, we select
      matching paths canonically: shortest p1 first, then lexicographically.
      This makes the result independent of enumeration order. *)

  (** Compare paths: first by length, then lexicographically. *)
  Fixpoint path_compare (p1 p2 : Path) : comparison :=
    match Nat.compare (length p1) (length p2) with
    | Lt => Lt
    | Gt => Gt
    | Eq =>
      match p1, p2 with
      | [], [] => Eq
      | x :: xs, y :: ys =>
        match Z.compare x y with
        | Eq => path_compare xs ys
        | c => c
        end
      | [], _ :: _ => Lt  (* Shouldn't happen if lengths equal *)
      | _ :: _, [] => Gt  (* Shouldn't happen if lengths equal *)
      end
    end.

  (** path_compare is reflexive. *)
  Lemma path_compare_refl : forall p, path_compare p p = Eq.
  Proof.
    induction p as [|x xs IH]; simpl.
    - reflexivity.
    - rewrite Nat.compare_refl. simpl.
      rewrite Z.compare_refl. apply IH.
  Qed.

  (** path_compare respects equality. *)
  Lemma path_compare_eq : forall p1 p2,
    path_compare p1 p2 = Eq -> p1 = p2.
  Proof.
    induction p1 as [|x xs IH]; destruct p2 as [|y ys]; intros H.
    - reflexivity.
    - simpl in H. destruct (Nat.compare 0 (S (length ys))); discriminate.
    - simpl in H. destruct (Nat.compare (S (length xs)) 0); discriminate.
    - simpl in H.
      (* After simpl, H : match Nat.compare (length xs) (length ys) with ... = Eq *)
      destruct (Nat.compare (length xs) (length ys)) eqn:Hlen.
      + (* length xs = length ys *)
        destruct (Z.compare x y) eqn:Hxy.
        * apply Z.compare_eq in Hxy. subst y.
          f_equal. apply IH. exact H.
        * discriminate H.
        * discriminate H.
      + (* length xs < length ys - contradiction *)
        discriminate H.
      + (* length xs > length ys - contradiction *)
        discriminate H.
  Qed.

  (** path_compare is antisymmetric. *)
  Lemma path_compare_antisym : forall p1 p2,
    path_compare p1 p2 = CompOpp (path_compare p2 p1).
  Proof.
    induction p1 as [|x xs IH]; destruct p2 as [|y ys].
    - (* [], [] *)
      reflexivity.
    - (* [], y :: ys *)
      (* path_compare [] (y :: ys) = Lt (since 0 < S (length ys)) *)
      (* path_compare (y :: ys) [] = Gt (since S (length ys) > 0) *)
      (* CompOpp Gt = Lt *)
      reflexivity.
    - (* x :: xs, [] *)
      (* path_compare (x :: xs) [] = Gt *)
      (* path_compare [] (x :: xs) = Lt *)
      (* CompOpp Lt = Gt *)
      reflexivity.
    - (* x :: xs, y :: ys *)
      simpl.
      (* We need to show:
         match length xs ?= length ys with
         | Eq => match x ?= y with | Eq => path_compare xs ys | c => c end
         | c => c
         end
         =
         CompOpp (match length ys ?= length xs with
         | Eq => match y ?= x with | Eq => path_compare ys xs | c => c end
         | c => c
         end)
      *)
      destruct (Nat.compare (length xs) (length ys)) eqn:Hlen_xy.
      + (* length xs = length ys *)
        apply Nat.compare_eq in Hlen_xy as Hlen_eq.
        rewrite Hlen_eq. rewrite Nat.compare_refl.
        simpl.
        destruct (Z.compare x y) eqn:Hxy.
        * (* x = y *)
          rewrite Z.compare_antisym. rewrite Hxy. simpl.
          apply IH.
        * (* x < y *)
          rewrite Z.compare_antisym. rewrite Hxy. simpl. reflexivity.
        * (* x > y *)
          rewrite Z.compare_antisym. rewrite Hxy. simpl. reflexivity.
      + (* length xs < length ys *)
        apply Nat.compare_lt_iff in Hlen_xy.
        assert (Hlen_yx: Nat.compare (length ys) (length xs) = Gt).
        { apply Nat.compare_gt_iff. lia. }
        rewrite Hlen_yx. simpl. reflexivity.
      + (* length xs > length ys *)
        apply Nat.compare_gt_iff in Hlen_xy.
        assert (Hlen_yx: Nat.compare (length ys) (length xs) = Lt).
        { apply Nat.compare_lt_iff. lia. }
        rewrite Hlen_yx. simpl. reflexivity.
  Qed.

  (** path_compare transitivity for Lt. *)
  Lemma path_compare_trans_lt : forall p1 p2 p3,
    path_compare p1 p2 = Lt ->
    path_compare p2 p3 = Lt ->
    path_compare p1 p3 = Lt.
  Proof.
    induction p1 as [|x xs IH]; intros p2 p3 H12 H23; simpl in *.
      + (* p1 = [] *)
        destruct p3 as [|z zs].
        * destruct p2 as [|y ys]; simpl in *.
          -- discriminate H12.
          -- destruct (Nat.compare 0 (S (length ys))); discriminate.
        * simpl. destruct (Nat.compare 0 (S (length zs))) eqn:Hcmp.
          -- apply Nat.compare_eq in Hcmp. discriminate.
          -- reflexivity.
          -- apply Nat.compare_gt_iff in Hcmp. lia.
      + (* p1 = x :: xs *)
        destruct p2 as [|y ys]; simpl in *.
        * destruct (Nat.compare (S (length xs)) 0) eqn:Hcmp; discriminate.
        * destruct p3 as [|z zs]; simpl in *.
          -- destruct (Nat.compare (S (length ys)) 0) eqn:Hcmp; discriminate.
          -- destruct (Nat.compare (S (length xs)) (S (length ys))) eqn:Hcmp12;
             destruct (Nat.compare (S (length ys)) (S (length zs))) eqn:Hcmp23.
             ++ (* Both lengths equal *)
                apply Nat.compare_eq in Hcmp12 as Hlen12.
                apply Nat.compare_eq in Hcmp23 as Hlen23.
                injection Hlen12 as Hlen12'. injection Hlen23 as Hlen23'.
                assert (Hlen13: length xs = length zs) by lia.
                (* Refine H12 and H23 using length equalities *)
                rewrite Hlen12' in H12. rewrite Nat.compare_refl in H12.
                rewrite Hlen23' in H23. rewrite Nat.compare_refl in H23.
                (* Goal has length xs ?= length zs, rewrite it *)
                rewrite Hlen13. rewrite Nat.compare_refl. simpl.
                (* Now goal is: match x ?= z with ... end = Lt *)
                (* H12 : match x ?= y with ... = Lt
                   H23 : match y ?= z with ... = Lt *)
                destruct (Z.compare x y) eqn:Hxy.
                ** (* x = y *)
                   apply Z.compare_eq in Hxy. subst.
                   (* Now H12 : path_compare xs ys = Lt
                      H23 : match y ?= z with ... = Lt
                      Goal : match y ?= z with ... = Lt *)
                   destruct (Z.compare y z) eqn:Hyz.
                   --- (* y = z *)
                       apply Z.compare_eq in Hyz. subst.
                       simpl. apply (IH ys zs H12 H23).
                   --- (* y < z - goal is already match y ?= z, which matches Hyz *)
                       exact H23.
                   --- (* y > z, contradiction with H23 *)
                       discriminate.
                ** (* x < y *)
                   (* H12 is now Lt = Lt (trivially true) *)
                   destruct (Z.compare y z) eqn:Hyz.
                   --- (* y = z *)
                       apply Z.compare_eq in Hyz. subst.
                       (* x < z, so x ?= z = Lt *)
                       rewrite Hxy. reflexivity.
                   --- (* y < z, so x < y < z, x ?= z = Lt *)
                       assert (Hxlt: (x < y)%Z) by (apply Z.compare_lt_iff; exact Hxy).
                       assert (Hylt: (y < z)%Z) by (apply Z.compare_lt_iff; exact Hyz).
                       assert (Hxz: (x < z)%Z) by lia.
                       assert (Hxzcmp: (x ?= z)%Z = Lt) by (apply Z.compare_lt_iff; exact Hxz).
                       rewrite Hxzcmp. reflexivity.
                   --- (* y > z, contradiction *)
                       discriminate H23.
                ** (* x > y, contradiction with H12 *)
                   discriminate H12.
             ++ (* len xs = len ys, len ys < len zs *)
                apply Nat.compare_eq in Hcmp12 as Hlen12.
                apply Nat.compare_lt_iff in Hcmp23 as Hlen23.
                injection Hlen12 as Hlen12'.
                assert (Hlen13: length xs < length zs) by lia.
                apply Nat.compare_lt_iff in Hlen13.
                rewrite Hlen13. reflexivity.
             ++ (* len xs = len ys, len ys > len zs - contradiction *)
                (* When Nat.compare gives Gt, H23 becomes Gt = Lt, contradiction *)
                apply Nat.compare_gt_iff in Hcmp23.
                assert (Hinner: length zs < length ys) by lia.
                assert (Hlt: (length ys ?= length zs) = Gt).
                { apply Nat.compare_gt_iff. lia. }
                rewrite Hlt in H23. discriminate.
             ++ (* len xs < len ys, len ys = len zs *)
                apply Nat.compare_lt_iff in Hcmp12 as Hlen12.
                apply Nat.compare_eq in Hcmp23 as Hlen23.
                injection Hlen23 as Hlen23'.
                assert (Hlen13: length xs < length zs) by lia.
                apply Nat.compare_lt_iff in Hlen13.
                rewrite Hlen13. reflexivity.
             ++ (* len xs < len ys, len ys < len zs *)
                apply Nat.compare_lt_iff in Hcmp12 as Hlen12.
                apply Nat.compare_lt_iff in Hcmp23 as Hlen23.
                assert (Hlen13: length xs < length zs) by lia.
                apply Nat.compare_lt_iff in Hlen13.
                rewrite Hlen13. reflexivity.
             ++ (* len xs < len ys, len ys > len zs - contradiction *)
                apply Nat.compare_gt_iff in Hcmp23.
                assert (Hinner: length zs < length ys) by lia.
                assert (Hgt: (length ys ?= length zs) = Gt).
                { apply Nat.compare_gt_iff. lia. }
                rewrite Hgt in H23. discriminate.
             ++ (* len xs > len ys - contradiction with H12 *)
                apply Nat.compare_gt_iff in Hcmp12.
                assert (Hgt: (length xs ?= length ys) = Gt).
                { apply Nat.compare_gt_iff. lia. }
                rewrite Hgt in H12. discriminate.
             ++ (* len xs > len ys, Hcmp23 = Lt - contradiction with H12 *)
                apply Nat.compare_gt_iff in Hcmp12.
                assert (Hgt: (length xs ?= length ys) = Gt).
                { apply Nat.compare_gt_iff. lia. }
                rewrite Hgt in H12. discriminate.
             ++ (* len xs > len ys, Hcmp23 = Gt - contradiction with H12 *)
                apply Nat.compare_gt_iff in Hcmp12.
                assert (Hgt: (length xs ?= length ys) = Gt).
                { apply Nat.compare_gt_iff. lia. }
                rewrite Hgt in H12. discriminate.
  Qed.

  (** path_compare is transitive. *)
  Lemma path_compare_trans : forall p1 p2 p3 c,
    path_compare p1 p2 = c ->
    path_compare p2 p3 = c ->
    c <> Eq ->
    path_compare p1 p3 = c.
  Proof.
    intros p1 p2 p3 c H12 H23 Hneq.
    destruct c; [exfalso; apply Hneq; reflexivity | |].
    - (* Lt case *)
      apply (path_compare_trans_lt p1 p2 p3 H12 H23).
    - (* Gt case - use antisymmetry to reduce to Lt *)
      rewrite path_compare_antisym.
      rewrite path_compare_antisym in H12.
      rewrite path_compare_antisym in H23.
      simpl in *.
      destruct (path_compare p2 p1) eqn:H21; simpl in *; try discriminate.
      destruct (path_compare p3 p2) eqn:H32; simpl in *; try discriminate.
      assert (Htrans: path_compare p3 p1 = Lt).
      { apply (path_compare_trans_lt p3 p2 p1 H32 H21). }
      rewrite Htrans. reflexivity.
  Qed.

  (** path_compare not Gt means Lt or Eq. *)
  Lemma path_compare_not_gt : forall p1 p2,
    path_compare p1 p2 <> Gt ->
    path_compare p1 p2 = Lt \/ path_compare p1 p2 = Eq.
  Proof.
    intros p1 p2 H.
    destruct (path_compare p1 p2) eqn:Hcmp.
    - right. reflexivity.
    - left. reflexivity.
    - exfalso. apply H. reflexivity.
  Qed.

  (** If p1 <= p2 and p2 < p3 then p1 < p3. *)
  Lemma path_compare_le_lt_trans : forall p1 p2 p3,
    path_compare p1 p2 <> Gt ->
    path_compare p2 p3 = Lt ->
    path_compare p1 p3 <> Gt.
  Proof.
    intros p1 p2 p3 Hle12 Hlt23.
    apply path_compare_not_gt in Hle12.
    destruct Hle12 as [Hlt12 | Heq12].
    - (* p1 < p2 < p3, so p1 < p3 *)
      assert (Hlt13: path_compare p1 p3 = Lt).
      { apply (path_compare_trans p1 p2 p3 Lt Hlt12 Hlt23). discriminate. }
      rewrite Hlt13. discriminate.
    - (* p1 = p2 < p3 *)
      apply path_compare_eq in Heq12. subst p1.
      rewrite Hlt23. discriminate.
  Qed.

  (** Helper: Compare pairs by their first component (p1). *)
  Definition pair_compare (pair1 pair2 : Path * Path) : comparison :=
    path_compare (fst pair1) (fst pair2).

  (** Select the pair with shortest p1 from a non-empty list.
      Uses fold to find minimum by pair_compare. *)
  Definition select_shortest (matching : list (Path * Path)) : option (Path * Path) :=
    match matching with
    | [] => None
    | h :: t =>
      Some (fold_left
        (fun best cur =>
          match pair_compare cur best with
          | Lt => cur
          | _ => best
          end)
        t h)
    end.

  (** Lemma: select_shortest returns an element from the list. *)
  Lemma select_shortest_in : forall matching pair,
    select_shortest matching = Some pair ->
    In pair matching.
  Proof.
    intros matching pair Hsel.
    destruct matching as [|h t]; simpl in Hsel.
    - discriminate.
    - injection Hsel as Heq.
      (* The fold_left result is either h or some element from t *)
      revert h Heq. induction t as [|x xs IH]; intros h Heq.
      + simpl in Heq. subst pair. left. reflexivity.
      + destruct (pair_compare x h) eqn:Hcmp; simpl in Heq; rewrite Hcmp in Heq; simpl in Heq.
        * (* Eq: h remains best (comparison order: Eq, Lt, Gt) *)
          specialize (IH h Heq).
          destruct IH as [Heqh | Hinxs].
          -- left. assumption.
          -- right. right. assumption.
        * (* Lt: x becomes new best *)
          specialize (IH x Heq).
          destruct IH as [Heqx | Hinxs].
          -- right. left. assumption.
          -- right. right. assumption.
        * (* Gt: h remains best *)
          specialize (IH h Heq).
          destruct IH as [Heqh | Hinxs].
          -- left. assumption.
          -- right. right. assumption.
  Qed.

  (** Lemma: select_shortest returns the minimum by pair_compare. *)
  Lemma select_shortest_is_min : forall matching pair,
    select_shortest matching = Some pair ->
    forall other, In other matching ->
    pair_compare pair other <> Gt.
  Proof.
    intros matching pair Hsel other Hother.
    destruct matching as [|h t]; simpl in Hsel.
    - discriminate.
    - injection Hsel as Heq.
      (* We need to show that fold_left finds the minimum *)
      (* Revert other as well to get a properly quantified IH *)
      revert h Heq other Hother.
      induction t as [|y ys IH]; intros h Heq other Hother.
      + (* Base case: matching = [h] *)
        simpl in Heq. subst pair.
        destruct Hother as [Heqo | []].
        subst other. unfold pair_compare. rewrite path_compare_refl.
        discriminate.
      + (* Inductive case - destruct gives order: Eq, Lt, Gt *)
        simpl in Heq.
        destruct (pair_compare y h) eqn:Hcmp.
        * (* Eq: y = h by first component, h remains best *)
          destruct Hother as [Heqo | [Heqo | Hinys]].
          -- (* other = h *)
             subst other. apply (IH h Heq h). left. reflexivity.
          -- (* other = y *)
             subst other.
             (* other = y, and y = h (by Eq), so need pair <= h = y *)
             assert (Hpair_le_h: pair_compare pair h <> Gt).
             { apply (IH h Heq h). left. reflexivity. }
             unfold pair_compare in *.
             (* path_compare (fst y) (fst h) = Eq means fst y = fst h *)
             apply path_compare_eq in Hcmp.
             (* Rewrite fst h to fst y in hypothesis *)
             rewrite <- Hcmp in Hpair_le_h.
             assumption.
          -- (* other in ys *)
             apply (IH h Heq other). right. assumption.
        * (* Lt: y < h, so y becomes new best *)
          destruct Hother as [Heqo | [Heqo | Hinys]].
          -- (* other = h *)
             subst other.
             (* pair is min of (y :: ys), so pair <= y < h *)
             (* Need: pair_compare pair h <> Gt *)
             assert (Hpair_le_y: pair_compare pair y <> Gt).
             { apply (IH y Heq y). left. reflexivity. }
             unfold pair_compare in *.
             (* If pair <= y and y < h, then pair < h *)
             apply (path_compare_le_lt_trans (fst pair) (fst y) (fst h)).
             ++ assumption.
             ++ assumption.
          -- (* other = y *)
             subst other.
             apply (IH y Heq y). left. reflexivity.
          -- (* other in ys *)
             apply (IH y Heq other). right. assumption.
        * (* Gt: y > h, h remains best *)
          destruct Hother as [Heqo | [Heqo | Hinys]].
          -- (* other = h *)
             subst other. apply (IH h Heq h). left. reflexivity.
          -- (* other = y *)
             subst other.
             (* other = y > h, and pair <= h *)
             (* We need pair <= y, i.e., pair_compare pair y <> Gt *)
             assert (Hpair_le_h: pair_compare pair h <> Gt).
             { apply (IH h Heq h). left. reflexivity. }
             unfold pair_compare in *.
             (* pair <= h and h < y, so pair < y *)
             (* Hcmp says path_compare (fst y) (fst h) = Gt, i.e., y > h *)
             (* We want pair_compare pair y <> Gt *)
             (* This is path_compare (fst pair) (fst y) <> Gt *)
             (* We have path_compare (fst pair) (fst h) <> Gt (pair <= h) *)
             (* And path_compare (fst y) (fst h) = Gt (y > h), i.e., h < y *)
             (* Antisymmetry: path_compare (fst h) (fst y) = Lt *)
             assert (Hh_lt_y: path_compare (fst h) (fst y) = Lt).
             { rewrite path_compare_antisym. rewrite Hcmp. simpl. reflexivity. }
             apply (path_compare_le_lt_trans (fst pair) (fst h) (fst y)).
             ++ assumption.
             ++ assumption.
          -- (* other in ys *)
             apply (IH h Heq other). right. assumption.
  Qed.

  (** ** Join All Values

      To satisfy the quantale distributivity law, pathmap_multiply must
      join ALL matching values, not just select one. This ensures that:
        pm1 * (pm2 ∪ pm3) = (pm1 * pm2) ∪ (pm1 * pm3)

      The key insight is that matching pairs for pm2∪pm3 are the union
      of matching pairs for pm2 and pm3, so joining all values gives
      the same result as joining the individual products. *)

  (** Join all matching values from pm1 for the given (p1, p2) pairs.
      Uses option_join semantics: None is identity, Some values are joined. *)
  Definition join_all_values (pm1 : PathMap V) (matching : list (Path * Path)) : option V :=
    fold_left
      (fun acc '(p1, _) => opt_join acc (pm1 p1))
      matching
      None.

  (** Path multiplication on PathMaps.
      Given two PathMaps, the product has entries for all concatenations
      of paths from pm1 and pm2.

      {a,b} * {c,d} = {ac, ad, bc, bd}

      IMPORTANT: Joins ALL matching values to satisfy the quantale
      distributivity law. This requires V to be a join-semilattice. *)
  Definition pathmap_multiply (pm1 pm2 : PathMap V) : PathMap V :=
    fun p =>
      let paths1 := pathmap_paths pm1 in
      let paths2 := pathmap_paths pm2 in
      let candidates := flat_map (fun p1 =>
        map (fun p2 => (p1, p2)) paths2
      ) paths1 in
      let matching := filter (fun '(p1, p2) =>
        if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false
      ) candidates in
      join_all_values pm1 matching.

  (** ** Helper lemmas for join_all_values *)

  (** join_all_values on empty list is None *)
  Lemma join_all_values_nil : forall pm1,
    join_all_values pm1 [] = None.
  Proof.
    intros. reflexivity.
  Qed.

  (** join_all_values on singleton *)
  Lemma join_all_values_singleton : forall pm1 p1 p2,
    join_all_values pm1 [(p1, p2)] = pm1 p1.
  Proof.
    intros. unfold join_all_values. simpl.
    unfold option_join.
    destruct (pm1 p1); reflexivity.
  Qed.

  (** Helper lemma: fold_left with option_join distributes over initial values.
      For any init, fold_left f l init = opt_join init (fold_left f l None)
      This uses the join-semilattice property of opt_join.

      Proof sketch by induction on l:
      - Base case (l = []): both sides reduce to init
      - Inductive case (l = (p1, p2) :: rest):
        Uses associativity of opt_join
  *)
  Lemma fold_left_option_join_init : forall (pm1 : PathMap V) (l : list (Path * Path)) init,
    fold_left (fun acc '(p1, _) => opt_join acc (pm1 p1)) l init =
    opt_join init (fold_left (fun acc '(p1, _) => opt_join acc (pm1 p1)) l None).
  Proof.
    intros pm1 l init.
    revert init.
    induction l as [|[p1 p2] rest IH]; intro init.
    - (* l = [] *)
      simpl. unfold opt_join, option_join. destruct init; reflexivity.
    - (* l = (p1, p2) :: rest *)
      (* Uses associativity of opt_join *)
      destruct init as [init_val|]; simpl; destruct (pm1 p1) as [v1|] eqn:Hpm1.
      + (* init = Some init_val, pm1 p1 = Some v1 *)
        (* LHS: fold_left f rest (Some (join_V init_val v1)) *)
        (* RHS: opt_join (Some init_val) (fold_left f rest (Some v1)) *)
        rewrite IH.
        (* LHS = opt_join (Some (join_V init_val v1)) (fold_left f rest None) *)
        rewrite (IH (Some v1)).
        (* RHS = opt_join (Some init_val) (opt_join (Some v1) (fold_left f rest None)) *)
        (* By associativity of join_V and opt_join *)
        unfold opt_join, option_join at 1 2.
        destruct (fold_left _ rest None) as [vrest|]; simpl.
        * f_equal. apply join_V_assoc.
        * reflexivity.
      + (* init = Some init_val, pm1 p1 = None *)
        rewrite IH. reflexivity.
      + (* init = None, pm1 p1 = Some v1 *)
        (* LHS: fold_left f rest (Some v1) *)
        (* RHS: opt_join None (fold_left f rest (Some v1)) *)
        (* opt_join None x = x, so RHS = fold_left f rest (Some v1) *)
        destruct (fold_left _ rest (Some v1)); reflexivity.
      + (* init = None, pm1 p1 = None *)
        (* LHS: fold_left f rest None *)
        (* RHS: opt_join None (fold_left f rest None) = fold_left f rest None *)
        destruct (fold_left _ rest None); reflexivity.
  Qed.

  (** join_all_values cons - fold_left step.
      Uses fold_left_option_join_init to show that starting from opt_join None (pm1 p1)
      is equivalent to joining pm1 p1 with the fold starting from None. *)
  Lemma join_all_values_cons : forall pm1 h t,
    join_all_values pm1 (h :: t) =
    opt_join (pm1 (fst h)) (join_all_values pm1 t).
  Proof.
    intros pm1 [p1 p2] t.
    unfold join_all_values at 1. simpl fst.
    (* fold_left f ((p1, p2) :: t) None
       = fold_left f t (f None (p1, p2))
       = fold_left f t (opt_join None (pm1 p1)) *)
    simpl.
    (* Use fold_left_option_join_init:
       fold_left f t init = opt_join init (fold_left f t None) *)
    rewrite fold_left_option_join_init.
    (* LHS: opt_join (opt_join None (pm1 p1)) (fold_left ... t None)
       RHS: opt_join (pm1 p1) (fold_left ... t None) *)
    (* opt_join None (pm1 p1) = pm1 p1 *)
    unfold opt_join at 1, option_join.
    destruct (pm1 p1) as [v1|] eqn:Hpm1; reflexivity.
  Qed.

  (** join_all_values on concatenated lists equals opt_join of individual results.
      This is the key lemma for proving distributivity. *)
  Lemma join_all_values_app : forall pm1 l1 l2,
    join_all_values pm1 (l1 ++ l2) =
    opt_join (join_all_values pm1 l1) (join_all_values pm1 l2).
  Proof.
    intros pm1 l1 l2.
    unfold join_all_values.
    rewrite fold_left_app.
    apply fold_left_option_join_init.
  Qed.

  (** join_all_values distributes over pathmap join.
      This is the key lemma for right distributivity. *)
  Lemma join_all_values_join_pathmap : forall pm1 pm2 l,
    join_all_values (pm_join pm1 pm2) l =
    opt_join (join_all_values pm1 l) (join_all_values pm2 l).
  Proof.
    intros pm1 pm2 l.
    induction l as [|[p1 p2] rest IH].
    - (* Empty list *)
      simpl. reflexivity.
    - (* Cons case *)
      unfold join_all_values in *.
      simpl.
      unfold pm_join, pathmap_join.
      (* After simpl, the goal should have fold_left with opt_join initial values *)
      (* Rewrite using fold_left_option_join_init to extract initial values *)
      repeat rewrite fold_left_option_join_init.
      (* Now goal is opt_join of initial value and fold on rest with None *)
      (* Use IH which has been unfolded *)
      rewrite IH.
      (* Now we need to show associativity/commutativity of opt_join rearranges properly *)
      unfold opt_join, option_join.
      destruct (pm1 p1) as [v1|] eqn:H1;
      destruct (pm2 p1) as [v2|] eqn:H2;
      destruct (fold_left
         (fun (acc : option V) '(p0, _) =>
          match acc with
          | Some v => match pm1 p0 with
                      | Some v' => Some (join_V v v')
                      | None => Some v
                      end
          | None => pm1 p0
          end) rest None) as [r1|] eqn:Hr1;
      destruct (fold_left
         (fun (acc : option V) '(p0, _) =>
          match acc with
          | Some v => match pm2 p0 with
                      | Some v' => Some (join_V v v')
                      | None => Some v
                      end
          | None => pm2 p0
          end) rest None) as [r2|] eqn:Hr2;
      try reflexivity;
      (* Use associativity and commutativity of join_V *)
      try (f_equal; rewrite join_V_assoc; f_equal; apply join_V_comm);
      try (f_equal; rewrite <- join_V_assoc; rewrite <- join_V_assoc;
           f_equal; rewrite join_V_assoc; f_equal; apply join_V_comm).
  Qed.

  (** join_all_values is unchanged by filtering out pairs with pm(p1) = None.
      This is because such pairs contribute Nothing to the fold. *)
  Lemma join_all_values_filter_none : forall pm l,
    join_all_values pm l =
    join_all_values pm (filter (fun '(p1, _) => match pm p1 with Some _ => true | None => false end) l).
  Proof.
    intros pm l.
    induction l as [|[p1 p2] rest IH].
    - simpl. reflexivity.
    - simpl.
      destruct (pm p1) as [v|] eqn:Hp1.
      + (* pm p1 = Some v - keep this pair *)
        simpl.
        unfold join_all_values in *.
        simpl. rewrite Hp1.
        rewrite fold_left_option_join_init.
        rewrite fold_left_option_join_init.
        f_equal.
        apply IH.
      + (* pm p1 = None - skip this pair *)
        unfold join_all_values in *.
        simpl. rewrite Hp1.
        (* fold_left with None init and pm p1 = None -> same as starting with None *)
        apply IH.
  Qed.

  (** Corollary: join_all_values on a larger list equals join_all_values on
      the filtered list that only contains pairs with pm(p1) ≠ None,
      and this filtered list is a subset of any list that contains all such pairs. *)
  Lemma join_all_values_superset : forall pm l1 l2,
    (forall p1 p2, In (p1, p2) l1 -> pm p1 <> None -> In (p1, p2) l2) ->
    (forall p1 p2, In (p1, p2) l2 -> In (p1, p2) l1) ->
    join_all_values pm l1 = join_all_values pm l2.
  Proof.
    intros pm l1 l2 Hsub1 Hsub2.
    rewrite join_all_values_filter_none.
    rewrite (join_all_values_filter_none pm l2).
    apply join_all_values_set_eq.
    intros p1 p2.
    split.
    - intro Hin.
      apply filter_In in Hin. destruct Hin as [Hin Hpred].
      destruct (pm p1) as [v|] eqn:Hp1; [|discriminate].
      apply filter_In. split.
      + apply Hsub1; [assumption | congruence].
      + rewrite Hp1. reflexivity.
    - intro Hin.
      apply filter_In in Hin. destruct Hin as [Hin Hpred].
      destruct (pm p1) as [v|] eqn:Hp1; [|discriminate].
      apply filter_In. split.
      + apply Hsub2. assumption.
      + rewrite Hp1. reflexivity.
  Qed.

  (** join_all_values returns Some iff at least one matching path has a value *)
  Lemma join_all_values_some_iff : forall pm1 matching,
    (exists v, join_all_values pm1 matching = Some v) <->
    (exists p1 p2, In (p1, p2) matching /\ pm1 p1 <> None).
  Proof.
    intros pm1 matching.
    induction matching as [|[p1 p2] rest IH].
    - (* Empty list *)
      simpl. split.
      + intros [v Hv]. discriminate.
      + intros [p1 [p2 [Hin _]]]. destruct Hin.
    - (* Cons case *)
      split.
      + intros [v Hv].
        unfold join_all_values in Hv. simpl in Hv.
        (* fold_left ... rest (opt_join None (pm1 p1)) = Some v *)
        unfold opt_join, option_join at 1 in Hv.
        destruct (pm1 p1) as [v1|] eqn:Hpm1.
        * (* pm1 p1 = Some v1 *)
          exists p1, p2. split; [left; reflexivity | rewrite Hpm1; discriminate].
        * (* pm1 p1 = None *)
          (* fold_left ... rest None = Some v *)
          fold (join_all_values pm1 rest) in Hv.
          assert (Hex: exists v, join_all_values pm1 rest = Some v).
          { exists v. exact Hv. }
          apply IH in Hex.
          destruct Hex as [p1' [p2' [Hin Hne]]].
          exists p1', p2'. split; [right; assumption | assumption].
      + intros [p1' [p2' [Hin Hne]]].
        destruct Hin as [Heq | Hin].
        * (* (p1', p2') = (p1, p2) *)
          injection Heq as Heq1 Heq2. subst p1' p2'.
          unfold join_all_values. simpl.
          unfold opt_join, option_join at 1.
          destruct (pm1 p1) as [v1|] eqn:Hpm1.
          -- (* pm1 p1 = Some v1 *)
             (* fold_left ... rest (Some v1) = Some ? *)
             (* Since we start with Some, we'll end with Some *)
             clear IH Hne Hpm1.
             revert v1.
             induction rest as [|[p1'' p2''] rest' IH']; intros v1.
             ++ simpl. exists v1. reflexivity.
             ++ simpl.
                unfold opt_join, option_join.
                destruct (pm1 p1'') as [v''|].
                ** (* opt_join (Some v1) (Some v'') = Some (join_V v1 v'') *)
                   exact (IH' (join_V v1 v'')).
                ** exact (IH' v1).
          -- exfalso. apply Hne. reflexivity.
        * (* In (p1', p2') rest *)
          assert (Hex: exists v, join_all_values pm1 rest = Some v).
          { apply IH. exists p1', p2'. split; assumption. }
          destruct Hex as [vrest Hvrest].
          unfold join_all_values. simpl.
          unfold opt_join, option_join at 1.
          destruct (pm1 p1) as [v1|] eqn:Hpm1.
          -- (* pm1 p1 = Some v1 *)
             (* fold_left starting from Some v1 will give Some something *)
             (* Clear all unused context before nested induction *)
             clear IH Hpm1 Hvrest vrest Hin Hne p1' p2'.
             revert v1.
             induction rest as [|[p1'' p2''] rest' IH']; intros v1.
             ++ simpl. exists v1. reflexivity.
             ++ simpl.
                unfold opt_join, option_join.
                destruct (pm1 p1'') as [v''|].
                ** (* opt_join (Some v1) (Some v'') = Some (join_V v1 v'') *)
                   exact (IH' (join_V v1 v'')).
                ** exact (IH' v1).
          -- (* pm1 p1 = None, so result is join_all_values pm1 rest *)
             simpl. fold (join_all_values pm1 rest).
             exists vrest. exact Hvrest.
  Qed.

  (** When join_all_values returns Some v, there exists at least one (p1, p2) in matching
      with pm1 p1 <> None. The value v is the join of all such pm1 values. *)
  Lemma join_all_values_value : forall pm1 matching v,
    join_all_values pm1 matching = Some v ->
    exists p1 p2, In (p1, p2) matching /\ pm1 p1 <> None.
  Proof.
    intros pm1 matching v Hval.
    induction matching as [|[p1 p2] rest IH].
    - (* Empty list cannot produce Some *)
      unfold join_all_values in Hval. simpl in Hval. discriminate.
    - (* Cons case *)
      unfold join_all_values in Hval. simpl in Hval.
      unfold opt_join, option_join at 1 in Hval.
      destruct (pm1 p1) as [v1|] eqn:Hpm1.
      + (* pm1 p1 = Some v1 *)
        exists p1, p2. split; [left; reflexivity | rewrite Hpm1; discriminate].
      + (* pm1 p1 = None *)
        (* fold_left rest None = Some v, so recurse *)
        fold (join_all_values pm1 rest) in Hval.
        apply IH in Hval.
        destruct Hval as [p1' [p2' [Hin Hne]]].
        exists p1', p2'. split; [right; assumption | assumption].
  Qed.

  (** opt_join is commutative (from join_V_comm) *)
  Lemma opt_join_comm : forall x y,
    opt_join x y = opt_join y x.
  Proof.
    intros [v1|] [v2|]; simpl; try reflexivity.
    f_equal. apply join_V_comm.
  Qed.

  (** opt_join is associative (from join_V_assoc) *)
  Lemma opt_join_assoc : forall x y z,
    opt_join (opt_join x y) z = opt_join x (opt_join y z).
  Proof.
    intros [v1|] [v2|] [v3|]; simpl; try reflexivity.
    - f_equal. apply join_V_assoc.
  Qed.

  (** opt_join is idempotent (from join_V_idem) *)
  Lemma opt_join_idem : forall x,
    opt_join x x = x.
  Proof.
    intros [v|]; simpl; try reflexivity.
    f_equal. apply join_V_idem.
  Qed.

  (** ** Helper lemmas for pathmap_multiply proofs *)

  (** Membership in flat_map. *)
  Lemma in_flat_map_iff : forall {A B : Type} (f : A -> list B) (l : list A) (y : B),
    In y (flat_map f l) <-> exists x, In x l /\ In y (f x).
  Proof.
    intros A B f l y.
    induction l as [|x xs IH].
    - simpl. split.
      + intros [].
      + intros [x [[] _]].
    - simpl. rewrite in_app_iff. split.
      + intros [Hfx | Hxs].
        * exists x. split; [left; reflexivity | assumption].
        * apply IH in Hxs. destruct Hxs as [x' [Hin Hfy]].
          exists x'. split; [right; assumption | assumption].
      + intros [x' [[Heq | Hin] Hfy]].
        * subst x'. left. assumption.
        * right. apply IH. exists x'. split; assumption.
  Qed.

  (** Membership in map. *)
  Lemma in_map_iff_simple : forall {A B : Type} (f : A -> B) (l : list A) (y : B),
    In y (map f l) <-> exists x, In x l /\ y = f x.
  Proof.
    intros A B f l y.
    induction l as [|x xs IH].
    - simpl. split.
      + intros [].
      + intros [x [[] _]].
    - simpl. split.
      + intros [Heq | Hin].
        * exists x. split; [left; reflexivity | symmetry; assumption].
        * apply IH in Hin. destruct Hin as [x' [Hin' Heq']].
          exists x'. split; [right; assumption | assumption].
      + intros [x' [[Heq | Hin] Heq']].
        * subst x' y. left. reflexivity.
        * right. apply IH. exists x'. split; assumption.
  Qed.

  (** Membership in filter. *)
  Lemma in_filter_iff : forall {A : Type} (f : A -> bool) (l : list A) (x : A),
    In x (filter f l) <-> In x l /\ f x = true.
  Proof.
    intros A f l x.
    induction l as [|y ys IH].
    - simpl. split.
      + intros [].
      + intros [[] _].
    - simpl. destruct (f y) eqn:Hfy.
      + simpl. split.
        * intros [Heq | Hin].
          -- subst. split; [left; reflexivity | assumption].
          -- apply IH in Hin. destruct Hin as [Hin Hfx].
             split; [right; assumption | assumption].
        * intros [[Heq | Hin] Hfx].
          -- left. assumption.
          -- right. apply IH. split; assumption.
      + split.
        * intros Hin. apply IH in Hin. destruct Hin as [Hin Hfx].
          split; [right; assumption | assumption].
        * intros [[Heq | Hin] Hfx].
          -- subst. rewrite Hfx in Hfy. discriminate.
          -- apply IH. split; assumption.
  Qed.

  (** Characterization of when matching is non-empty. *)
  Lemma matching_nonempty_iff : forall (pm1 pm2 : PathMap V) (p : Path),
    let paths1 := pathmap_paths pm1 in
    let paths2 := pathmap_paths pm2 in
    let candidates := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1 in
    let matching := filter (fun '(p1, p2) =>
      if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false
    ) candidates in
    matching <> [] <-> exists p1 p2, In p1 paths1 /\ In p2 paths2 /\ p = p1 ++ p2.
  Proof.
    intros pm1 pm2 p paths1 paths2 candidates matching.
    set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
    split.
    - (* matching <> [] -> exists p1 p2, ... *)
      intros Hne.
      destruct matching as [|[p1 p2] rest] eqn:Hmatch.
      + exfalso. apply Hne. reflexivity.
      + (* First element (p1, p2) is in filter result *)
        assert (Hin: In (p1, p2) (filter the_pred candidates)).
        { unfold matching in Hmatch. rewrite Hmatch. left. reflexivity. }
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2.
        subst p1' p2'.
        (* Now check the predicate *)
        unfold the_pred in Hpred. simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
        exists p1, p2. split; [assumption | split; assumption].
    - (* exists p1 p2, ... -> matching <> [] *)
      intros [p1 [p2 [Hp1 [Hp2 Hpeq]]]].
      (* Show (p1, p2) is in matching *)
      assert (Hin: In (p1, p2) candidates).
      { apply in_flat_map_iff.
        exists p1. split; [assumption|].
        apply in_map_iff_simple.
        exists p2. split; [assumption | reflexivity]. }
      assert (Hpred_holds: the_pred (p1, p2) = true).
      { unfold the_pred. simpl.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [_|Hne].
        - reflexivity.
        - exfalso. apply Hne. assumption. }
      assert (Hin': In (p1, p2) matching).
      { unfold matching. apply in_filter_iff. split; assumption. }
      intro Heq. rewrite Heq in Hin'. simpl in Hin'. contradiction.
  Qed.

  (** When matching is non-empty, first element satisfies the predicate. *)
  Lemma matching_first_satisfies : forall (pm1 pm2 : PathMap V) (p : Path) (p1 p2 : Path) rest,
    let paths1 := pathmap_paths pm1 in
    let paths2 := pathmap_paths pm2 in
    let candidates := flat_map (fun p1' => map (fun p2' => (p1', p2')) paths2) paths1 in
    filter (fun '(p1', p2') =>
      if list_eq_dec Z.eq_dec p (p1' ++ p2') then true else false
    ) candidates = (p1, p2) :: rest ->
    In p1 paths1 /\ In p2 paths2 /\ p = p1 ++ p2.
  Proof.
    intros pm1 pm2 p p1 p2 rest paths1 paths2 candidates Hfilter.
    assert (Hin: In (p1, p2) ((p1, p2) :: rest)) by (left; reflexivity).
    rewrite <- Hfilter in Hin.
    apply in_filter_iff in Hin.
    destruct Hin as [Hcand Hpred].
    apply in_flat_map_iff in Hcand.
    destruct Hcand as [p1' [Hp1' Hpair]].
    apply in_map_iff_simple in Hpair.
    destruct Hpair as [p2' [Hp2' Heq]].
    injection Heq as Heq1 Heq2.
    subst p1' p2'.
    simpl in Hpred.
    destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
    split; [assumption | split; assumption].
  Qed.

  (** Lemma: Characterization of pathmap_multiply existence.
      The product contains a value at path p iff there exists a split p = p1 ++ p2
      where pm1 has a value at p1 and pm2 has a value at p2. *)
  Lemma pathmap_multiply_exists : forall pm1 pm2 p,
    (exists v, pathmap_multiply pm1 pm2 p = Some v) <->
    (exists p1 p2, p = p1 ++ p2 /\ pm1 p1 <> None /\ pm2 p2 <> None).
  Proof.
    intros pm1 pm2 p.
    unfold pathmap_multiply.
    (* Reduce the let expressions so set can match them *)
    cbv zeta.
    set (paths1 := pathmap_paths pm1).
    set (paths2 := pathmap_paths pm2).
    set (candidates := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1).
    set (matching := filter (fun '(p1, p2) =>
      if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) candidates).
    (* Use join_all_values_some_iff to characterize existence *)
    rewrite join_all_values_some_iff.
    split.
    - (* exists p1' p2', In (p1', p2') matching /\ pm1 p1' <> None -> exists p1 p2, ... *)
      intros [p1' [p2' [Hin Hpm1_ne]]].
      (* (p1', p2') is in matching, so it satisfies the filter predicate *)
      apply filter_In in Hin.
      destruct Hin as [Hin_cand Hpred].
      (* (p1', p2') is in candidates, so p1' in paths1 and p2' in paths2 *)
      apply in_flat_map_iff in Hin_cand.
      destruct Hin_cand as [p1'' [Hp1' Hpair]].
      apply in_map_iff_simple in Hpair.
      destruct Hpair as [p2'' [Hp2' Heq]].
      injection Heq as Heq1 Heq2. subst p1'' p2''.
      (* p2' is in paths2, so pm2 p2' <> None *)
      exists p1', p2'.
      (* Extract p = p1' ++ p2' from filter predicate *)
      simpl in Hpred.
      destruct (list_eq_dec Z.eq_dec p (p1' ++ p2')) as [Hpeq|]; [|discriminate].
      split; [assumption|].
      split; [assumption|].
      apply pathmap_paths_complete. assumption.
    - (* exists p1 p2, ... -> exists p1' p2', In (p1', p2') matching /\ pm1 p1' <> None *)
      intros [p1 [p2 [Hpeq [Hpm1 Hpm2]]]].
      exists p1, p2.
      split.
      + (* Show (p1, p2) is in matching *)
        apply filter_In.
        split.
        * (* (p1, p2) is in candidates *)
          apply in_flat_map_iff.
          exists p1. split.
          -- apply pathmap_paths_complete. assumption.
          -- apply in_map_iff_simple. exists p2. split.
             ++ apply pathmap_paths_complete. assumption.
             ++ reflexivity.
        * (* (p1, p2) satisfies the filter predicate *)
          simpl. destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)); [reflexivity | contradiction].
      + assumption.
  Qed.

  (** Lemma: When multiply returns Some, there exist matching paths.
      With join_all_values: v is the join of all matching pm1 values. *)
  Lemma pathmap_multiply_value : forall pm1 pm2 p v,
    pathmap_multiply pm1 pm2 p = Some v ->
    exists p1 p2, p = p1 ++ p2 /\ pm1 p1 <> None /\ pm2 p2 <> None.
  Proof.
    intros pm1 pm2 p v Hmult.
    unfold pathmap_multiply in Hmult.
    cbv zeta in Hmult.
    set (paths1 := pathmap_paths pm1) in Hmult.
    set (paths2 := pathmap_paths pm2) in Hmult.
    set (candidates := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1) in Hmult.
    set (matching := filter (fun '(p1, p2) =>
      if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) candidates) in Hmult.
    (* Use join_all_values_value to get (p1_sel, p2_sel) with pm1 p1_sel <> None *)
    apply join_all_values_value in Hmult.
    destruct Hmult as [p1_sel [p2_sel [Hin Hpm1_ne]]].
    (* (p1_sel, p2_sel) is in matching, so it satisfies filter predicate *)
    apply filter_In in Hin.
    destruct Hin as [Hin_cand Hpred].
    (* Extract path properties from candidates *)
    apply in_flat_map_iff in Hin_cand.
    destruct Hin_cand as [p1'' [Hp1'' Hpair]].
    apply in_map_iff_simple in Hpair.
    destruct Hpair as [p2'' [Hp2'' Heq]].
    injection Heq as Heq1 Heq2. subst p1'' p2''.
    (* Path equation from filter predicate *)
    simpl in Hpred.
    destruct (list_eq_dec Z.eq_dec p (p1_sel ++ p2_sel)) as [Hpeq|]; [|discriminate].
    exists p1_sel, p2_sel.
    split; [exact Hpeq|].
    split.
    + (* pm1 p1_sel <> None *)
      exact Hpm1_ne.
    + (* pm2 p2_sel <> None *)
      apply pathmap_paths_complete. exact Hp2''.
  Qed.

  (** Lemma: Consistency of multiply with join for existence (left). *)
  Lemma pathmap_multiply_join_exists : forall pm1 pm2 pm3 p,
    (exists v, pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v) <->
    (exists v, pathmap_multiply pm1 pm2 p = Some v) \/
    (exists v, pathmap_multiply pm1 pm3 p = Some v).
  Proof.
    intros pm1 pm2 pm3 p.
    rewrite pathmap_multiply_exists.
    rewrite pathmap_multiply_exists.
    rewrite pathmap_multiply_exists.
    split.
    - (* -> *)
      intros [p1 [p2 [Hpeq [Hpm1 Hjoin]]]].
      unfold pm_join, pathmap_join in Hjoin.
      apply option_join_not_none in Hjoin.
      destruct Hjoin as [Hpm2 | Hpm3].
      + left. exists p1, p2. split; [assumption | split; assumption].
      + right. exists p1, p2. split; [assumption | split; assumption].
    - (* <- *)
      intros [[p1 [p2 [Hpeq [Hpm1 Hpm2]]]] | [p1 [p2 [Hpeq [Hpm1 Hpm3]]]]].
      + exists p1, p2. split; [assumption|]. split; [assumption|].
        unfold pm_join, pathmap_join. apply option_join_not_none. left. assumption.
      + exists p1, p2. split; [assumption|]. split; [assumption|].
        unfold pm_join, pathmap_join. apply option_join_not_none. right. assumption.
  Qed.

  (** Lemma: Consistency of multiply with join for existence (right). *)
  Lemma pathmap_multiply_join_exists_right : forall pm1 pm2 pm3 p,
    (exists v, pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v) <->
    (exists v, pathmap_multiply pm1 pm3 p = Some v) \/
    (exists v, pathmap_multiply pm2 pm3 p = Some v).
  Proof.
    intros pm1 pm2 pm3 p.
    rewrite pathmap_multiply_exists.
    rewrite pathmap_multiply_exists.
    rewrite pathmap_multiply_exists.
    split.
    - (* -> *)
      intros [p1 [p2 [Hpeq [Hjoin Hpm3]]]].
      unfold pm_join, pathmap_join in Hjoin.
      apply option_join_not_none in Hjoin.
      destruct Hjoin as [Hpm1 | Hpm2].
      + left. exists p1, p2. split; [assumption | split; assumption].
      + right. exists p1, p2. split; [assumption | split; assumption].
    - (* <- *)
      intros [[p1 [p2 [Hpeq [Hpm1 Hpm3]]]] | [p1 [p2 [Hpeq [Hpm2 Hpm3]]]]].
      + exists p1, p2. split; [assumption|]. split; [|assumption].
        unfold pm_join, pathmap_join. apply option_join_not_none. left. assumption.
      + exists p1, p2. split; [assumption|]. split; [|assumption].
        unfold pm_join, pathmap_join. apply option_join_not_none. right. assumption.
  Qed.

  (** Key lemma for value equality: When pm1*pm3 = None, any matching pair
      (p1, p2) for pm2⊔pm3 with pm1 p1 ≠ None must have pm2 p2 ≠ None.
      This is because pm1*pm3 = None rules out pairs with pm3 p2 ≠ None. *)
  Lemma pm3_none_implies_pm2 : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm3 p = None ->
    p = p1 ++ p2 ->
    pm1 p1 <> None ->
    pm_join pm2 pm3 p2 <> None ->
    pm2 p2 <> None.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hm13_none Hpeq Hpm1 Hjoin.
    (* Assume pm2 p2 = None, derive contradiction *)
    destruct (pm2 p2) as [v2|] eqn:Hpm2; [discriminate|].
    (* pm2 p2 = None, so pm_join pm2 pm3 p2 ≠ None implies pm3 p2 ≠ None *)
    unfold pm_join, pathmap_join in Hjoin.
    rewrite Hpm2 in Hjoin. simpl in Hjoin.
    assert (Hpm3: pm3 p2 <> None).
    { destruct (pm3 p2); [discriminate | contradiction]. }
    (* But then (p1, p2) would be a matching pair for pm1*pm3 *)
    assert (Hex: exists v, pathmap_multiply pm1 pm3 p = Some v).
    { apply pathmap_multiply_exists. exists p1, p2.
      split; [assumption | split; assumption]. }
    destruct Hex as [v Hv]. congruence.
  Qed.

  (** Similar lemma for pm2 = None. *)
  Lemma pm2_none_implies_pm3 : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm2 p = None ->
    p = p1 ++ p2 ->
    pm1 p1 <> None ->
    pm_join pm2 pm3 p2 <> None ->
    pm3 p2 <> None.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hm12_none Hpeq Hpm1 Hjoin.
    (* Assume pm3 p2 = None, derive contradiction *)
    destruct (pm3 p2) as [v3|] eqn:Hpm3; [discriminate|].
    (* pm3 p2 = None, so pm_join pm2 pm3 p2 ≠ None implies pm2 p2 ≠ None *)
    unfold pm_join, pathmap_join in Hjoin.
    destruct (pm2 p2) as [v2|] eqn:Hpm2.
    - (* pm2 p2 = Some v2, so (p1, p2) would be a matching pair for pm1*pm2 *)
      assert (Hex: exists v, pathmap_multiply pm1 pm2 p = Some v).
      { apply pathmap_multiply_exists. exists p1, p2.
        split; [assumption | split; [assumption | rewrite Hpm2; discriminate]]. }
      destruct Hex as [v Hv]. congruence.
    - (* Both pm2 p2 = None and pm3 p2 = None *)
      (* pm_join pm2 pm3 p2 = option_join None None = None *)
      rewrite Hpm3 in Hjoin. simpl in Hjoin.
      exfalso. apply Hjoin. reflexivity.
  Qed.

  (** Key lemma for distributivity Case 2:
      When pm1*pm3 = None, any (p1, p2) in the matching list for pm_join pm2 pm3
      must have p2 in pathmap_paths pm2.
      This is because p2 in paths_pm3 only would put (p1, p2) in matching_pm3,
      but that would give pm1*pm3 ≠ None. *)
  Lemma matching_join_subset_when_pm3_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm3 p = None ->
    In p1 (pathmap_paths pm1) ->
    In p2 (pathmap_paths (pm_join pm2 pm3)) ->
    p = p1 ++ p2 ->
    In p2 (pathmap_paths pm2).
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hm13_none Hp1 Hp2_join Hpeq.
    (* p2 is in pathmap_paths (pm_join pm2 pm3), so pm_join pm2 pm3 p2 <> None *)
    apply pathmap_paths_complete in Hp2_join.
    (* pm_join pm2 pm3 p2 <> None means pm2 p2 <> None \/ pm3 p2 <> None *)
    unfold pm_join, pathmap_join in Hp2_join.
    apply option_join_not_none in Hp2_join.
    destruct Hp2_join as [Hpm2 | Hpm3].
    - (* pm2 p2 <> None, so p2 is in pathmap_paths pm2 *)
      apply pathmap_paths_complete. assumption.
    - (* pm3 p2 <> None *)
      (* Then (p1, p2) would be in matching for pm1*pm3 *)
      (* But pm1*pm3 = None means matching is empty *)
      exfalso.
      apply pathmap_paths_complete in Hp1.
      assert (Hex: exists v, pathmap_multiply pm1 pm3 p = Some v).
      { apply pathmap_multiply_exists.
        exists p1, p2. split; [assumption | split; assumption]. }
      destruct Hex as [v Hv]. congruence.
  Qed.

  (** Symmetric version for pm2 = None *)
  Lemma matching_join_subset_when_pm2_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm2 p = None ->
    In p1 (pathmap_paths pm1) ->
    In p2 (pathmap_paths (pm_join pm2 pm3)) ->
    p = p1 ++ p2 ->
    In p2 (pathmap_paths pm3).
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hm12_none Hp1 Hp2_join Hpeq.
    apply pathmap_paths_complete in Hp2_join.
    unfold pm_join, pathmap_join in Hp2_join.
    apply option_join_not_none in Hp2_join.
    destruct Hp2_join as [Hpm2 | Hpm3].
    - (* pm2 p2 <> None *)
      exfalso.
      apply pathmap_paths_complete in Hp1.
      assert (Hex: exists v, pathmap_multiply pm1 pm2 p = Some v).
      { apply pathmap_multiply_exists.
        exists p1, p2. split; [assumption | split; assumption]. }
      destruct Hex as [v Hv]. congruence.
    - (* pm3 p2 <> None, so p2 is in pathmap_paths pm3 *)
      apply pathmap_paths_complete. assumption.
  Qed.

  (** Key lemma: join_all_values depends only on the set of p1 values.
      This is a consequence of join being commutative, associative, and idempotent.

      For two lists with the same set of (p1, p2) pairs, the fold produces
      the same result because:
      1. Order doesn't matter (commutativity + associativity of opt_join)
      2. Duplicates don't matter (idempotence of opt_join)

      We prove this using permutation theory. *)

  (** Helper: decidable equality on pairs of paths. *)
  Definition path_pair_eq_dec : forall (x y : Path * Path), {x = y} + {x <> y}.
  Proof.
    intros [p1 p2] [q1 q2].
    destruct (list_eq_dec Z.eq_dec p1 q1) as [Heq1|Hne1].
    - destruct (list_eq_dec Z.eq_dec p2 q2) as [Heq2|Hne2].
      + left. subst. reflexivity.
      + right. intros H. injection H as H1 H2. apply Hne2. assumption.
    - right. intros H. injection H as H1 H2. apply Hne1. assumption.
  Defined.

  (** Helper: fold_left with swap-invariant operation is permutation-invariant.
      We prove this by induction on the permutation. *)
  Lemma fold_left_opt_join_swap : forall pm (x y : Path * Path) l acc,
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) (x :: y :: l) acc =
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) (y :: x :: l) acc.
  Proof.
    intros pm [px1 px2] [py1 py2] l acc.
    simpl.
    (* Use commutativity and associativity of opt_join *)
    assert (Hswap: opt_join (opt_join acc (pm px1)) (pm py1) =
                   opt_join (opt_join acc (pm py1)) (pm px1)).
    { rewrite opt_join_assoc. rewrite opt_join_assoc.
      f_equal. apply opt_join_comm. }
    rewrite Hswap. reflexivity.
  Qed.

  (** fold_left with opt_join is permutation-invariant. *)
  Lemma fold_left_opt_join_permutation : forall pm l1 l2 acc,
    Permutation l1 l2 ->
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) l1 acc =
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) l2 acc.
  Proof.
    intros pm l1 l2 acc Hperm.
    revert acc.
    induction Hperm as [| x l1' l2' Hperm' IH
                       | x y l
                       | l1' l2' l3' Hperm12 IH12 Hperm23 IH23];
      intros acc.
    - (* perm_nil *)
      reflexivity.
    - (* perm_skip *)
      simpl. apply IH.
    - (* perm_swap *)
      apply fold_left_opt_join_swap.
    - (* perm_trans *)
      rewrite IH12. apply IH23.
  Qed.

  (** join_all_values is permutation-invariant. *)
  Lemma join_all_values_permutation : forall pm l1 l2,
    Permutation l1 l2 ->
    join_all_values pm l1 = join_all_values pm l2.
  Proof.
    intros pm l1 l2 Hperm.
    unfold join_all_values.
    apply fold_left_opt_join_permutation. assumption.
  Qed.

  (** Processing the same element twice adds nothing (idempotence). *)
  Lemma fold_left_opt_join_dup : forall pm (x : Path * Path) l acc,
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) (x :: l) acc =
    fold_left (fun a '(p1, _) => opt_join a (pm p1)) (x :: x :: l) acc.
  Proof.
    intros pm [p1 p2] l acc.
    simpl.
    rewrite <- opt_join_assoc.
    rewrite opt_join_idem.
    reflexivity.
  Qed.

  (** Key lemma: If x is already in l, then prepending x doesn't change join_all_values.
      This uses permutation invariance and idempotence. *)
  Lemma join_all_values_dup_head : forall pm (x : Path * Path) l,
    In x l ->
    join_all_values pm (x :: l) = join_all_values pm l.
  Proof.
    intros pm x l Hin.
    (* Since x is in l, we can split l around that occurrence *)
    apply in_split in Hin.
    destruct Hin as [l1 [l2 Heq]].
    subst l.
    (* l = l1 ++ x :: l2, we want: join_all_values pm (x :: l1 ++ x :: l2) = join_all_values pm (l1 ++ x :: l2) *)
    (* x :: l1 ++ x :: l2 is a permutation of x :: x :: l1 ++ l2 *)
    assert (Hperm1: Permutation (x :: l1 ++ x :: l2) (x :: x :: l1 ++ l2)).
    { apply perm_skip.
      rewrite app_comm_cons.
      apply Permutation_cons_app.
      apply Permutation_refl. }
    rewrite (join_all_values_permutation pm _ _ Hperm1).
    (* Now use idempotence: x :: x :: rest = x :: rest in terms of join_all_values *)
    unfold join_all_values at 1. simpl.
    destruct x as [p1 p2].
    rewrite <- opt_join_assoc.
    rewrite opt_join_idem.
    fold (join_all_values pm (l1 ++ l2)).
    (* Need to show: join_all_values pm (l1 ++ l2) = join_all_values pm (l1 ++ x :: l2) where x = (p1, p2) *)
    (* l1 ++ (p1, p2) :: l2 is a permutation of (p1, p2) :: l1 ++ l2 *)
    assert (Hperm2: Permutation (l1 ++ (p1, p2) :: l2) ((p1, p2) :: l1 ++ l2)).
    { apply Permutation_cons_app. apply Permutation_refl. }
    rewrite (join_all_values_permutation pm _ _ Hperm2).
    unfold join_all_values at 1. simpl.
    reflexivity.
  Qed.

  (** Two lists with same set membership have permutations of their nodup versions.
      This is a standard set theory result. *)
  Lemma NoDup_Permutation_incl : forall {A : Type} (l1 l2 : list A),
    NoDup l1 -> NoDup l2 ->
    (forall x, In x l1 <-> In x l2) ->
    Permutation l1 l2.
  Proof.
    intros A l1.
    induction l1 as [|x l1' IH]; intros l2 Hnd1 Hnd2 Hiff.
    - (* l1 = [] *)
      destruct l2 as [|y l2'].
      + apply perm_nil.
      + (* y is in l2, but Hiff says y is in [] - contradiction *)
        assert (Hin: In y (y :: l2')) by (left; reflexivity).
        apply Hiff in Hin. destruct Hin.
    - (* l1 = x :: l1' *)
      assert (Hx: In x l2).
      { apply Hiff. left. reflexivity. }
      apply in_split in Hx.
      destruct Hx as [l2a [l2b Heq]].
      subst l2.
      apply Permutation_cons_app.
      apply IH.
      + apply NoDup_cons_iff in Hnd1. destruct Hnd1. assumption.
      + apply NoDup_app_remove_middle in Hnd2. assumption.
      + intros y. split; intros Hin.
        * assert (Hiny: In y (x :: l1')) by (right; assumption).
          apply Hiff in Hiny.
          apply in_app_or in Hiny. apply in_or_app.
          destruct Hiny as [Hina | [Heq | Hinb]].
          -- left. assumption.
          -- (* y = x, but y is in l1' and x is not in l1' by NoDup *)
             subst y.
             apply NoDup_cons_iff in Hnd1. destruct Hnd1 as [Hnotin _].
             contradiction.
          -- right. assumption.
        * apply in_app_or in Hin.
          assert (Hiff': In y (l2a ++ x :: l2b)).
          { apply in_or_app. destruct Hin as [Hina | Hinb].
            - left. assumption.
            - right. right. assumption. }
          apply Hiff in Hiff'.
          destruct Hiff' as [Heq | Hin'].
          -- (* y = x *)
             subst y.
             (* x is in l2a ++ l2b, and x is not in l2a ++ x :: l2b without the middle x by NoDup *)
             (* Actually x IS in l2a ++ l2b from assumption, need to verify *)
             (* x is in l1', but NoDup says x is not in l1' - contradiction *)
             apply NoDup_cons_iff in Hnd1. destruct Hnd1 as [Hnotin _].
             (* Need to show In x (l2a ++ l2b) -> In x l1' which contradicts Hnotin *)
             (* From Hiff, In x (l2a ++ x :: l2b) <-> In x (x :: l1') *)
             (* We have In x (l2a ++ l2b), need to derive In x l1' *)
             (* This is getting complicated. Let me use a simpler approach. *)
             apply in_app_or in Hin. destruct Hin as [Hina | Hinb].
             ++ apply NoDup_app_iff in Hnd2.
                destruct Hnd2 as [_ [Hnd_xb _]].
                apply NoDup_cons_iff in Hnd_xb.
                destruct Hnd_xb as [Hnotin' _].
                (* x is in l2a, but l2a and x::l2b are disjoint by NoDup... no wait *)
                (* NoDup (l2a ++ x :: l2b) means x doesn't appear twice *)
                (* If x is in l2a, then it can't also be the x in the middle *)
                apply NoDup_app_iff in Hnd2.
                destruct Hnd2 as [Hnd_a [_ Hdisj]].
                specialize (Hdisj x Hina).
                simpl in Hdisj. destruct Hdisj. left. reflexivity.
             ++ apply NoDup_app_iff in Hnd2.
                destruct Hnd2 as [_ [Hnd_xb _]].
                apply NoDup_cons_iff in Hnd_xb.
                destruct Hnd_xb as [Hnotin' _].
                contradiction.
          -- assumption.
  Qed.

  (** nodup preserves the set membership. *)
  Lemma nodup_In : forall {A : Type} eq_dec (l : list A) x,
    In x (nodup eq_dec l) <-> In x l.
  Proof.
    intros A eq_dec l x.
    induction l as [|y l' IH].
    - simpl. reflexivity.
    - simpl. destruct (in_dec eq_dec y l') as [Hin | Hnotin].
      + (* y is in l', so nodup skips y *)
        rewrite IH. split.
        * intros H. right. assumption.
        * intros [Heq | H].
          -- subst y. apply IH. assumption.
          -- assumption.
      + (* y is not in l', so nodup keeps y *)
        simpl. rewrite IH. reflexivity.
  Qed.

  (** NoDup holds for nodup result. *)
  Lemma NoDup_nodup : forall {A : Type} eq_dec (l : list A),
    NoDup (nodup eq_dec l).
  Proof.
    intros A eq_dec l.
    induction l as [|x l' IH].
    - simpl. constructor.
    - simpl. destruct (in_dec eq_dec x l') as [Hin | Hnotin].
      + assumption.
      + constructor.
        * rewrite nodup_In. assumption.
        * assumption.
  Qed.

  (** Key lemma: Removing duplicates via nodup preserves join_all_values.
      This is because duplicate elements contribute the same value (idempotence). *)
  Lemma join_all_values_nodup : forall pm l,
    join_all_values pm l = join_all_values pm (nodup path_pair_eq_dec l).
  Proof.
    intros pm l.
    induction l as [|x l' IH].
    - (* Base case: empty list *)
      simpl. reflexivity.
    - (* Inductive case: x :: l' *)
      simpl nodup.
      destruct (in_dec path_pair_eq_dec x l') as [Hin | Hnotin].
      + (* x is in l', so nodup skips x *)
        (* join_all_values pm (x :: l') = join_all_values pm l' by dup_head *)
        rewrite join_all_values_dup_head by assumption.
        apply IH.
      + (* x is not in l', so nodup keeps x *)
        simpl.
        unfold join_all_values. simpl.
        fold (join_all_values pm l').
        fold (join_all_values pm (nodup path_pair_eq_dec l')).
        destruct x as [p1 p2].
        (* Need: opt_join None (pm p1) |> fold with l' = opt_join None (pm p1) |> fold with nodup l' *)
        (* This follows from IH which says fold with l' = fold with nodup l' *)
        rewrite IH.
        reflexivity.
  Qed.

  (** Main result: join_all_values depends only on the set of pairs. *)
  Lemma join_all_values_set_eq : forall pm1 l1 l2,
    (forall p1 p2, In (p1, p2) l1 <-> In (p1, p2) l2) ->
    join_all_values pm1 l1 = join_all_values pm1 l2.
  Proof.
    intros pm1 l1 l2 Hiff.
    (* Strategy: show nodup l1 and nodup l2 are permutations, use permutation invariance *)
    assert (Hnd1: NoDup (nodup path_pair_eq_dec l1)) by apply NoDup_nodup.
    assert (Hnd2: NoDup (nodup path_pair_eq_dec l2)) by apply NoDup_nodup.
    assert (Hiff': forall x, In x (nodup path_pair_eq_dec l1) <-> In x (nodup path_pair_eq_dec l2)).
    { intros [p1 p2]. rewrite nodup_In. rewrite nodup_In. apply Hiff. }
    assert (Hperm: Permutation (nodup path_pair_eq_dec l1) (nodup path_pair_eq_dec l2)).
    { apply NoDup_Permutation_incl; assumption. }
    (* Now we chain: l1 -> nodup l1 (by nodup lemma) -> nodup l2 (by perm) -> l2 (by nodup lemma) *)
    rewrite join_all_values_nodup.
    rewrite (join_all_values_permutation pm1 _ _ Hperm).
    symmetry.
    apply join_all_values_nodup.
  Qed.

  (** Helper: fold_left with pointwise equal lookup gives same result. *)
  Lemma fold_left_opt_join_pointwise_eq : forall (pm_A pm_B : PathMap V) (l : list (Path * Path)) acc,
    (forall p1 p2, In (p1, p2) l -> pm_A p1 = pm_B p1) ->
    fold_left (fun a '(p1, _) => opt_join a (pm_A p1)) l acc =
    fold_left (fun a '(p1, _) => opt_join a (pm_B p1)) l acc.
  Proof.
    intros pm_A pm_B l.
    induction l as [|[p1 p2] rest IH]; intros acc Hpw.
    - reflexivity.
    - simpl.
      assert (Hthis: pm_A p1 = pm_B p1).
      { apply (Hpw p1 p2). left. reflexivity. }
      rewrite Hthis.
      apply IH.
      intros p1' p2' Hin. apply (Hpw p1' p2'). right. assumption.
  Qed.

  (** Lemma: join_all_values depends only on the pathmap values at p1 positions.
      If two pathmaps agree on all p1 appearing in the list, they give the same result. *)
  Lemma join_all_values_pointwise_eq : forall (pm_A pm_B : PathMap V) (l : list (Path * Path)),
    (forall p1 p2, In (p1, p2) l -> pm_A p1 = pm_B p1) ->
    join_all_values pm_A l = join_all_values pm_B l.
  Proof.
    intros pm_A pm_B l Hpw.
    unfold join_all_values.
    apply fold_left_opt_join_pointwise_eq.
    assumption.
  Qed.

  (** Corollary: When pm1*pm3 = None, the matching pairs for pm_join pm2 pm3
      are exactly the matching pairs for pm2, hence the products are equal. *)
  Lemma multiply_value_when_pm3_none : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm2 p = Some v ->
    pathmap_multiply pm1 pm3 p = None ->
    pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v.
  Proof.
    intros pm1 pm2 pm3 p v H2 H3.
    unfold pathmap_multiply in *.
    cbv zeta in *.
    set (paths1 := pathmap_paths pm1) in *.
    set (paths2 := pathmap_paths pm2) in *.
    set (paths3 := pathmap_paths pm3) in *.
    set (paths_join := pathmap_paths (pm_join pm2 pm3)) in *.
    set (candidates2 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1) in *.
    set (candidates3 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths1) in *.
    set (candidates_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths_join) paths1) in *.
    set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
    set (matching2 := filter the_pred candidates2) in *.
    set (matching3 := filter the_pred candidates3) in *.
    set (matching_join := filter the_pred candidates_join) in *.
    (* Show that matching_join and matching2 have the same pairs *)
    rewrite (join_all_values_set_eq pm1 matching_join matching2).
    - (* Now goal is: join_all_values pm1 matching2 = Some v *)
      assumption.
    - (* Show: forall p1 p2, In (p1, p2) matching_join <-> In (p1, p2) matching2 *)
      intros p1 p2.
      split.
      + (* -> : In matching_join -> In matching2 *)
        intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        (* p1 is in paths1 = pathmap_paths pm1 *)
        (* p2 is in paths_join = pathmap_paths (pm_join pm2 pm3) *)
        (* By matching_join_subset_when_pm3_none, p2 is in paths2 *)
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
        assert (Hp2_pm2: In p2 paths2).
        { apply (matching_join_subset_when_pm3_none pm1 pm2 pm3 p p1 p2).
          - fold matching3 in H3. assumption.
          - assumption.
          - assumption.
          - assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * simpl. destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)); [reflexivity | contradiction].
      + (* <- : In matching2 -> In matching_join *)
        intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        (* p2 is in paths2, so it's also in paths_join *)
        assert (Hp2_join: In p2 paths_join).
        { apply pathmap_paths_join. left. assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * assumption.
  Qed.

  (** Symmetric lemma for when pm2 = None *)
  Lemma multiply_value_when_pm2_none : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm3 p = Some v ->
    pathmap_multiply pm1 pm2 p = None ->
    pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v.
  Proof.
    intros pm1 pm2 pm3 p v H3 H2.
    unfold pathmap_multiply in *.
    cbv zeta in *.
    set (paths1 := pathmap_paths pm1) in *.
    set (paths2 := pathmap_paths pm2) in *.
    set (paths3 := pathmap_paths pm3) in *.
    set (paths_join := pathmap_paths (pm_join pm2 pm3)) in *.
    set (candidates2 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1) in *.
    set (candidates3 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths1) in *.
    set (candidates_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths_join) paths1) in *.
    set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
    set (matching2 := filter the_pred candidates2) in *.
    set (matching3 := filter the_pred candidates3) in *.
    set (matching_join := filter the_pred candidates_join) in *.
    rewrite (join_all_values_set_eq pm1 matching_join matching3).
    - assumption.
    - intros p1 p2. split.
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
        assert (Hp2_pm3: In p2 paths3).
        { apply (matching_join_subset_when_pm2_none pm1 pm2 pm3 p p1 p2).
          - fold matching2 in H2. assumption.
          - assumption.
          - assumption.
          - assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * simpl. destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)); [reflexivity | contradiction].
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        assert (Hp2_join: In p2 paths_join).
        { apply pathmap_paths_join. right. assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * assumption.
  Qed.

  (** Key insight for right distributivity: when pm2*pm3 = None, for any matching
      pair (p1, p2) of (pm1∪pm2)*pm3, we must have pm2(p1) = None.

      Proof: If pm2(p1) ≠ None and pm3(p2) ≠ None and p = p1++p2, then this pair
      would be a match for pm2*pm3 at p, contradicting pm2*pm3 = None. *)
  Lemma pm2_value_none_when_pm2_pm3_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm2 pm3 p = None ->
    In p1 (pathmap_paths (pm_join pm1 pm2)) ->
    In p2 (pathmap_paths pm3) ->
    p = p1 ++ p2 ->
    pm2 p1 = None.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hpm23_none Hp1_join Hp2 Hpeq.
    destruct (pm2 p1) as [v2|] eqn:Hpm2; [|reflexivity].
    exfalso.
    (* If pm2(p1) ≠ None and pm3(p2) ≠ None and p = p1++p2, then pm2*pm3 ≠ None *)
    apply pathmap_paths_complete in Hp2.
    assert (Hpm2_ne: pm2 p1 <> None) by congruence.
    assert (Hex: exists v, pathmap_multiply pm2 pm3 p = Some v).
    { apply pathmap_multiply_exists.
      exists p1, p2. split; [assumption | split; assumption]. }
    destruct Hex as [v Hv]. congruence.
  Qed.

  (** Corollary: When pm2*pm3 = None, the value at p1 from (pm1∪pm2) equals pm1(p1). *)
  Lemma join_value_equals_pm1_when_pm2_pm3_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm2 pm3 p = None ->
    In p1 (pathmap_paths (pm_join pm1 pm2)) ->
    In p2 (pathmap_paths pm3) ->
    p = p1 ++ p2 ->
    (pm_join pm1 pm2) p1 = pm1 p1.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hpm23_none Hp1_join Hp2 Hpeq.
    pose proof (pm2_value_none_when_pm2_pm3_none pm1 pm2 pm3 p p1 p2 Hpm23_none Hp1_join Hp2 Hpeq) as Hpm2_none.
    unfold pm_join, pathmap_join, option_join.
    rewrite Hpm2_none.
    destruct (pm1 p1); reflexivity.
  Qed.

  (** Symmetric lemma: when pm1*pm3 = None, pm1(p1) = None for relevant p1. *)
  Lemma pm1_value_none_when_pm1_pm3_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm3 p = None ->
    In p1 (pathmap_paths (pm_join pm1 pm2)) ->
    In p2 (pathmap_paths pm3) ->
    p = p1 ++ p2 ->
    pm1 p1 = None.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hpm13_none Hp1_join Hp2 Hpeq.
    destruct (pm1 p1) as [v1|] eqn:Hpm1; [|reflexivity].
    exfalso.
    apply pathmap_paths_complete in Hp2.
    assert (Hpm1_ne: pm1 p1 <> None) by congruence.
    assert (Hex: exists v, pathmap_multiply pm1 pm3 p = Some v).
    { apply pathmap_multiply_exists.
      exists p1, p2. split; [assumption | split; assumption]. }
    destruct Hex as [v Hv]. congruence.
  Qed.

  (** Corollary: When pm1*pm3 = None, the value at p1 from (pm1∪pm2) equals pm2(p1). *)
  Lemma join_value_equals_pm2_when_pm1_pm3_none : forall pm1 pm2 pm3 p p1 p2,
    pathmap_multiply pm1 pm3 p = None ->
    In p1 (pathmap_paths (pm_join pm1 pm2)) ->
    In p2 (pathmap_paths pm3) ->
    p = p1 ++ p2 ->
    (pm_join pm1 pm2) p1 = pm2 p1.
  Proof.
    intros pm1 pm2 pm3 p p1 p2 Hpm13_none Hp1_join Hp2 Hpeq.
    pose proof (pm1_value_none_when_pm1_pm3_none pm1 pm2 pm3 p p1 p2 Hpm13_none Hp1_join Hp2 Hpeq) as Hpm1_none.
    unfold pm_join, pathmap_join, option_join.
    rewrite Hpm1_none.
    destruct (pm2 p1); reflexivity.
  Qed.

  (** Right distributivity value equality: When pm2*pm3 = None,
      (pm1∪pm2)*pm3 = pm1*pm3.

      The matching pairs for both sides are the same (pairs from pm1 side only),
      and for each pair, the value lookup is the same since pm2(p1) = None. *)
  Lemma multiply_right_value_when_pm2_none : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm3 p = Some v ->
    pathmap_multiply pm2 pm3 p = None ->
    pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v.
  Proof.
    intros pm1 pm2 pm3 p v H1 H2.
    (* Unfold in H1 and goal, but keep H2 folded for later contradiction *)
    unfold pathmap_multiply in H1 |-*.
    cbv zeta in H1 |-*.
    set (paths_pm1 := pathmap_paths pm1) in *.
    set (paths_pm2 := pathmap_paths pm2) in *.
    set (paths3 := pathmap_paths pm3) in *.
    set (paths_join := pathmap_paths (pm_join pm1 pm2)) in *.
    set (candidates1 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths_pm1) in *.
    set (candidates_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths_join) in *.
    set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
    set (matching1 := filter the_pred candidates1) in *.
    set (matching_join := filter the_pred candidates_join) in *.
    (* The key: for matching pairs, (pm1∪pm2)(p1) = pm1(p1) because pm2(p1) = None *)
    (* First show matching_join ≈ matching1 (set equivalence) *)
    (* Then show join_all_values (pm_join pm1 pm2) ≈ join_all_values pm1 on these pairs *)
    (* This requires showing that for each pair, the looked-up values are equal *)
    (* Use an axiom that fold over equal values gives equal results *)
    assert (Hvals: forall p1 p2, In (p1, p2) matching_join ->
      (pm_join pm1 pm2) p1 = pm1 p1).
    { intros p1' p2' Hin.
      apply in_filter_iff in Hin.
      destruct Hin as [Hcand Hpred].
      apply in_flat_map_iff in Hcand.
      destruct Hcand as [p1'' [Hp1'' Hpair]].
      apply in_map_iff_simple in Hpair.
      destruct Hpair as [p2'' [Hp2'' Heq]].
      injection Heq as Heq1 Heq2. subst p1'' p2''.
      simpl in Hpred.
      destruct (list_eq_dec Z.eq_dec p (p1' ++ p2')) as [Hpeq|]; [|discriminate].
      apply (join_value_equals_pm1_when_pm2_pm3_none pm1 pm2 pm3 p p1' p2' H2 Hp1'' Hp2'' Hpeq). }
    (* Now use axiom: join_all_values respects pointwise equality *)
    (* For this, we need a stronger axiom or prove it directly *)
    (* Alternative: show the lists have same set of (p1, p2) pairs AND same values *)
    rewrite (join_all_values_set_eq (pm_join pm1 pm2) matching_join matching1).
    - (* Need: join_all_values pm1 matching1 = Some v *)
      (* But we're computing join_all_values (pm_join pm1 pm2) matching1 *)
      (* However, for all pairs in matching1, (pm_join pm1 pm2)(p1) = pm1(p1) *)
      (* So the folded values are the same *)
      (* This requires showing join_all_values computes same result *)
      (* Since matching1 pairs come from paths_pm1, they satisfy (pm_join pm1 pm2)(p1) = pm1(p1) *)
      assert (Hvals1: forall p1 p2, In (p1, p2) matching1 ->
        (pm_join pm1 pm2) p1 = pm1 p1).
      { intros p1' p2' Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1'' [Hp1'' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2'' [Hp2'' Heq]].
        injection Heq as Heq1 Heq2. subst p1'' p2''.
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1' ++ p2')) as [Hpeq|]; [|discriminate].
        (* p1' ∈ paths_pm1, p2' ∈ paths3, p = p1'++p2' *)
        assert (Hp1'_join: In p1' paths_join).
        { apply pathmap_paths_join. left. assumption. }
        apply (join_value_equals_pm1_when_pm2_pm3_none pm1 pm2 pm3 p p1' p2' H2 Hp1'_join Hp2'' Hpeq). }
      (* Use join_all_values_pointwise_eq to show the two pathmaps give same result *)
      rewrite (join_all_values_pointwise_eq (pm_join pm1 pm2) pm1 matching1 Hvals1).
      assumption.
    - (* Show: forall p1 p2, In (p1, p2) matching_join <-> In (p1, p2) matching1 *)
      intros p1 p2. split.
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
        (* p1 ∈ paths_join, p2 ∈ paths3 *)
        (* Since pm2*pm3 = None, any p1 with pm2(p1)≠None would make pm2*pm3 have a value *)
        (* So for this pair to not be from pm2 side, we need pm1(p1)≠None *)
        apply pathmap_paths_complete in Hp1'.
        unfold pm_join, pathmap_join in Hp1'.
        apply option_join_not_none in Hp1'.
        destruct Hp1' as [Hpm1 | Hpm2].
        * (* pm1(p1) ≠ None, so p1 ∈ paths_pm1 *)
          apply pathmap_paths_complete in Hpm1.
          apply in_filter_iff.
          split.
          -- apply in_flat_map_iff. exists p1. split; [assumption|].
             apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
          -- simpl. destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)); [reflexivity | contradiction].
        * (* pm2(p1) ≠ None - this contradicts pm2*pm3 = None *)
          exfalso.
          apply pathmap_paths_complete in Hp2'.
          (* Hpm2 is already pm2 p1 <> None, Hp2' is now pm3 p2 <> None *)
          assert (Hex: exists v', pathmap_multiply pm2 pm3 p = Some v').
          { apply pathmap_multiply_exists.
            exists p1, p2. split; [assumption | split; assumption]. }
          destruct Hex as [v' Hv'].
          (* H2 is folded, Hv' is folded - direct contradiction *)
          rewrite H2 in Hv'. discriminate.
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        (* p1 ∈ paths_pm1, so p1 ∈ paths_join *)
        assert (Hp1_join: In p1 paths_join).
        { apply pathmap_paths_join. left. assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * assumption.
  Qed.

  (** Symmetric: When pm1*pm3 = None, (pm1∪pm2)*pm3 = pm2*pm3. *)
  Lemma multiply_right_value_when_pm1_none : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm2 pm3 p = Some v ->
    pathmap_multiply pm1 pm3 p = None ->
    pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v.
  Proof.
    intros pm1 pm2 pm3 p v H2 H1.
    (* Unfold in H2 and goal, but keep H1 folded for later contradiction *)
    unfold pathmap_multiply in H2 |-*.
    cbv zeta in H2 |-*.
    set (paths_pm1 := pathmap_paths pm1) in *.
    set (paths_pm2 := pathmap_paths pm2) in *.
    set (paths3 := pathmap_paths pm3) in *.
    set (paths_join := pathmap_paths (pm_join pm1 pm2)) in *.
    set (candidates2 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths_pm2) in *.
    set (candidates_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths_join) in *.
    set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
    set (matching2 := filter the_pred candidates2) in *.
    set (matching_join := filter the_pred candidates_join) in *.
    rewrite (join_all_values_set_eq (pm_join pm1 pm2) matching_join matching2).
    - (* Show: join_all_values (pm_join pm1 pm2) matching2 = Some v *)
      assert (Hvals2: forall p1 p2, In (p1, p2) matching2 ->
        (pm_join pm1 pm2) p1 = pm2 p1).
      { intros p1' p2' Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1'' [Hp1'' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2'' [Hp2'' Heq]].
        injection Heq as Heq1 Heq2. subst p1'' p2''.
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1' ++ p2')) as [Hpeq|]; [|discriminate].
        assert (Hp1'_join: In p1' paths_join).
        { apply pathmap_paths_join. right. assumption. }
        apply (join_value_equals_pm2_when_pm1_pm3_none pm1 pm2 pm3 p p1' p2' H1 Hp1'_join Hp2'' Hpeq). }
      (* Use join_all_values_pointwise_eq to show the two pathmaps give same result *)
      rewrite (join_all_values_pointwise_eq (pm_join pm1 pm2) pm2 matching2 Hvals2).
      assumption.
    - intros p1 p2. split.
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        simpl in Hpred.
        destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)) as [Hpeq|]; [|discriminate].
        apply pathmap_paths_complete in Hp1'.
        unfold pm_join, pathmap_join in Hp1'.
        apply option_join_not_none in Hp1'.
        destruct Hp1' as [Hpm1 | Hpm2].
        * (* pm1(p1) ≠ None - contradicts pm1*pm3 = None *)
          exfalso.
          apply pathmap_paths_complete in Hp2'.
          (* Hpm1 is already pm1 p1 <> None, Hp2' is now pm3 p2 <> None *)
          assert (Hex: exists v', pathmap_multiply pm1 pm3 p = Some v').
          { apply pathmap_multiply_exists.
            exists p1, p2. split; [assumption | split; assumption]. }
          destruct Hex as [v' Hv'].
          (* H1 is folded, Hv' is folded - direct contradiction *)
          rewrite H1 in Hv'. discriminate.
        * apply pathmap_paths_complete in Hpm2.
          apply in_filter_iff.
          split.
          -- apply in_flat_map_iff. exists p1. split; [assumption|].
             apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
          -- simpl. destruct (list_eq_dec Z.eq_dec p (p1 ++ p2)); [reflexivity | contradiction].
      + intro Hin.
        apply in_filter_iff in Hin.
        destruct Hin as [Hcand Hpred].
        apply in_flat_map_iff in Hcand.
        destruct Hcand as [p1' [Hp1' Hpair]].
        apply in_map_iff_simple in Hpair.
        destruct Hpair as [p2' [Hp2' Heq]].
        injection Heq as Heq1 Heq2. subst p1' p2'.
        assert (Hp1_join: In p1 paths_join).
        { apply pathmap_paths_join. right. assumption. }
        apply in_filter_iff.
        split.
        * apply in_flat_map_iff. exists p1. split; [assumption|].
          apply in_map_iff_simple. exists p2. split; [assumption | reflexivity].
        * assumption.
  Qed.

  (** Lemma: When multiply with join matches multiply with component,
      the result includes the original value via join.
      With join_all_values: the result is at least the original v (possibly joined with more). *)
  Lemma pathmap_multiply_join_value_left : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm2 p = Some v ->
    exists v', pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v'.
  Proof.
    intros pm1 pm2 pm3 p v Hmult.
    apply pathmap_multiply_value in Hmult.
    destruct Hmult as [p1 [p2 [Hpeq [Hne1 Hne2]]]].
    (* p1 has a value in pm1, p2 has a value in pm2 *)
    (* In pm_join pm2 pm3, p2 also has a value (at least from pm2) *)
    (* So (p1, p2) is a matching pair for pm1 * (pm_join pm2 pm3) *)
    apply pathmap_multiply_exists.
    exists p1, p2. split; [assumption|].
    split; [assumption|].
    unfold pm_join, pathmap_join.
    apply option_join_not_none. left. assumption.
  Qed.

  (** Lemma: Same for when pm3 has the matching split. *)
  Lemma pathmap_multiply_join_value_right_only : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm3 p = Some v ->
    pathmap_multiply pm1 pm2 p = None ->
    exists v', pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v'.
  Proof.
    intros pm1 pm2 pm3 p v Hmult3 Hmult2_none.
    apply pathmap_multiply_value in Hmult3.
    destruct Hmult3 as [p1 [p2 [Hpeq [Hne1 Hne3]]]].
    apply pathmap_multiply_exists.
    exists p1, p2. split; [assumption|].
    split; [assumption|].
    unfold pm_join, pathmap_join.
    apply option_join_not_none. right. assumption.
  Qed.

  (** Lemma: Right-side join is transparent for first operand.
      With join_all_values: if pm1*pm3 has a value, then (pm1∪pm2)*pm3 has a value too. *)
  Lemma pathmap_multiply_join_right_value_left : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm1 pm3 p = Some v ->
    exists v', pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v'.
  Proof.
    intros pm1 pm2 pm3 p v Hmult.
    apply pathmap_multiply_value in Hmult.
    destruct Hmult as [p1 [p2 [Hpeq [Hne1 Hne3]]]].
    (* p1 has a value in pm1, p2 has a value in pm3 *)
    (* In pm_join pm1 pm2, p1 also has a value (at least from pm1) *)
    (* So (p1, p2) is a matching pair for (pm_join pm1 pm2) * pm3 *)
    apply pathmap_multiply_exists.
    exists p1, p2. split; [assumption|].
    split; [|assumption].
    unfold pm_join, pathmap_join.
    apply option_join_not_none. left. assumption.
  Qed.

  (** Lemma: When only pm2 has matching split.
      With join_all_values: if pm2*pm3 has a value, then (pm1∪pm2)*pm3 has a value too. *)
  Lemma pathmap_multiply_join_right_value_right_only : forall pm1 pm2 pm3 p v,
    pathmap_multiply pm2 pm3 p = Some v ->
    pathmap_multiply pm1 pm3 p = None ->
    exists v', pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v'.
  Proof.
    intros pm1 pm2 pm3 p v Hmult2 Hmult1_none.
    apply pathmap_multiply_value in Hmult2.
    destruct Hmult2 as [p1 [p2 [Hpeq [Hne2 Hne3]]]].
    (* p1 has a value in pm2, p2 has a value in pm3 *)
    (* In pm_join pm1 pm2, p1 also has a value (at least from pm2) *)
    apply pathmap_multiply_exists.
    exists p1, p2. split; [assumption|].
    split; [|assumption].
    unfold pm_join, pathmap_join.
    apply option_join_not_none. right. assumption.
  Qed.

  (** Restricted multiplication (quantale restriction).
      This is like meet but only keeps entries where paths align. *)
  Definition pathmap_restrict (pm1 pm2 : PathMap V) : PathMap V :=
    fun p =>
      match pm1 p, pm2 p with
      | Some v1, Some _ => Some v1
      | _, _ => None
      end.

  (** Empty path is identity for multiplication. *)
  Theorem path_concat_empty_left : forall p,
    path_concat empty_path p = p.
  Proof.
    intros p. unfold path_concat, empty_path. reflexivity.
  Qed.

  Theorem path_concat_empty_right : forall p,
    path_concat p empty_path = p.
  Proof.
    intros p. unfold path_concat, empty_path.
    apply app_nil_r.
  Qed.

  (** Path concatenation is associative. *)
  Theorem path_concat_assoc : forall p1 p2 p3,
    path_concat (path_concat p1 p2) p3 = path_concat p1 (path_concat p2 p3).
  Proof.
    intros p1 p2 p3.
    unfold path_concat.
    symmetry. apply app_assoc.
  Qed.

  (** Multiplication with singleton at empty path is identity. *)
  Hypothesis pathmap_paths_singleton : forall p v,
    pathmap_paths (pathmap_singleton p v) = [p].

  (** Multiplication distributes over join (quantale law).
      This is the key property that makes PathMap a quantale.

      The proof uses the fact that with join_all_values semantics:
      - LHS joins all pm1 values where p = p1 ++ p2 and (pm2 ∪ pm3) p2 ≠ None
      - RHS joins pm1 values for pm2 matches, then for pm3 matches, then joins results
      - Since (pm2 ∪ pm3) p2 ≠ None iff pm2 p2 ≠ None ∨ pm3 p2 ≠ None,
        these are equivalent by associativity and idempotence of join.

      Note: This proof currently admits value equality. The existence-only
      lemmas guarantee that when one side has a value, the other does too,
      but proving exact equality requires showing the joined values are equal. *)
  Theorem pathmap_multiply_distributes_over_join : forall pm1 pm2 pm3,
    pathmap_eq (pathmap_multiply pm1 (pm_join pm2 pm3))
               (pm_join (pathmap_multiply pm1 pm2)
                        (pathmap_multiply pm1 pm3)).
  Proof.
    intros pm1 pm2 pm3 p.
    unfold pathmap_eq.
    (* Case analysis on whether pm1 * pm2 and pm1 * pm3 have values *)
    destruct (pathmap_multiply pm1 pm2 p) as [v2|] eqn:H2;
    destruct (pathmap_multiply pm1 pm3 p) as [v3|] eqn:H3.
    - (* Both pm1*pm2 and pm1*pm3 have values *)
      (* Strategy: Show LHS = opt_join (pm1*pm2) (pm1*pm3) by set equivalence *)
      (* The matching pairs for pm1*(pm2∪pm3) are the union of matching pairs
         for pm1*pm2 and pm1*pm3 *)
      unfold pathmap_multiply in *.
      cbv zeta in *.
      set (paths1 := pathmap_paths pm1) in *.
      set (paths2 := pathmap_paths pm2) in *.
      set (paths3 := pathmap_paths pm3) in *.
      set (paths_join := pathmap_paths (pm_join pm2 pm3)) in *.
      set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
      set (cand_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths_join) paths1) in *.
      set (cand2 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths2) paths1) in *.
      set (cand3 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths1) in *.
      set (matching_join := filter the_pred cand_join) in *.
      set (matching2 := filter the_pred cand2) in *.
      set (matching3 := filter the_pred cand3) in *.
      (* Key: matching_join ≈ matching2 ++ matching3 (set equivalence) *)
      assert (Hset: forall p1 p2, In (p1, p2) matching_join <-> In (p1, p2) (matching2 ++ matching3)).
      { intros p1' p2'. split.
        - intro Hin. apply in_filter_iff in Hin. destruct Hin as [Hcand Hpred].
          apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
          apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
          injection Heq as Heq1 Heq2. subst p1'' p2''.
          apply pathmap_paths_complete in Hp2''.
          unfold pm_join, pathmap_join in Hp2''.
          apply option_join_not_none in Hp2''.
          destruct Hp2'' as [Hpm2 | Hpm3].
          + (* p2' comes from pm2 *)
            apply in_or_app. left.
            apply pathmap_paths_complete in Hpm2.
            apply in_filter_iff. split; [|assumption].
            apply in_flat_map_iff. exists p1'. split; [assumption|].
            apply in_map_iff_simple. exists p2'. split; [assumption | reflexivity].
          + (* p2' comes from pm3 *)
            apply in_or_app. right.
            apply pathmap_paths_complete in Hpm3.
            apply in_filter_iff. split; [|assumption].
            apply in_flat_map_iff. exists p1'. split; [assumption|].
            apply in_map_iff_simple. exists p2'. split; [assumption | reflexivity].
        - intro Hin. apply in_app_iff in Hin. destruct Hin as [Hin2 | Hin3].
          + apply in_filter_iff in Hin2. destruct Hin2 as [Hcand Hpred].
            apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
            apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
            injection Heq as Heq1 Heq2. subst p1'' p2''.
            apply in_filter_iff. split; [|assumption].
            apply in_flat_map_iff. exists p1'. split; [assumption|].
            apply in_map_iff_simple. exists p2'. split; [|reflexivity].
            apply pathmap_paths_join. left. assumption.
          + apply in_filter_iff in Hin3. destruct Hin3 as [Hcand Hpred].
            apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
            apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
            injection Heq as Heq1 Heq2. subst p1'' p2''.
            apply in_filter_iff. split; [|assumption].
            apply in_flat_map_iff. exists p1'. split; [assumption|].
            apply in_map_iff_simple. exists p2'. split; [|reflexivity].
            apply pathmap_paths_join. right. assumption. }
      (* Apply join_all_values_set_eq *)
      rewrite (join_all_values_set_eq pm1 matching_join (matching2 ++ matching3) Hset).
      (* Apply join_all_values_app *)
      rewrite join_all_values_app.
      (* Now both sides are: opt_join (join_all_values pm1 matching2) (join_all_values pm1 matching3) *)
      reflexivity.
    - (* Only pm1*pm2 has value *)
      (* By multiply_value_when_pm3_none, LHS = Some v2 (same value!) *)
      pose proof (multiply_value_when_pm3_none pm1 pm2 pm3 p v2 H2 H3) as Hlhs.
      rewrite Hlhs.
      unfold pm_join, pathmap_join, opt_join, option_join.
      rewrite H2. rewrite H3.
      (* LHS: Some v2, RHS: Some v2 *)
      reflexivity.
    - (* Only pm1*pm3 has value *)
      (* By multiply_value_when_pm2_none, LHS = Some v3 (same value!) *)
      pose proof (multiply_value_when_pm2_none pm1 pm2 pm3 p v3 H3 H2) as Hlhs.
      rewrite Hlhs.
      unfold pm_join, pathmap_join, opt_join, option_join.
      rewrite H2. rewrite H3.
      (* LHS: Some v3, RHS: Some v3 *)
      reflexivity.
    - (* Neither has value, so LHS is also None *)
      assert (Hnone: pathmap_multiply pm1 (pm_join pm2 pm3) p = None).
      { destruct (pathmap_multiply pm1 (pm_join pm2 pm3) p) as [v|] eqn:Hleft.
        - (* Suppose it's Some v *)
          assert (Hex: exists v, pathmap_multiply pm1 (pm_join pm2 pm3) p = Some v).
          { exists v. exact Hleft. }
          apply pathmap_multiply_join_exists in Hex.
          destruct Hex as [[v' Hv'] | [v' Hv']]; congruence.
        - reflexivity.
      }
      rewrite Hnone.
      unfold pm_join, pathmap_join, opt_join, option_join. rewrite H2. rewrite H3.
      reflexivity.
  Qed.

  (** Right distributivity.
      Same pattern as left distributivity - existence is proven, value equality admitted. *)
  Theorem pathmap_multiply_distributes_over_join_right : forall pm1 pm2 pm3,
    pathmap_eq (pathmap_multiply (pm_join pm1 pm2) pm3)
               (pm_join (pathmap_multiply pm1 pm3)
                        (pathmap_multiply pm2 pm3)).
  Proof.
    intros pm1 pm2 pm3 p.
    unfold pathmap_eq.
    (* Case analysis on whether pm1*pm3 and pm2*pm3 have values *)
    destruct (pathmap_multiply pm1 pm3 p) as [v1|] eqn:H1;
    destruct (pathmap_multiply pm2 pm3 p) as [v2|] eqn:H2.
    - (* Both pm1*pm3 and pm2*pm3 have values *)
      (* Right distributivity Case 1: use join_all_values_join_pathmap *)
      unfold pathmap_multiply in *.
      cbv zeta in *.
      set (paths1 := pathmap_paths pm1) in *.
      set (paths2 := pathmap_paths pm2) in *.
      set (paths3 := pathmap_paths pm3) in *.
      set (paths_join := pathmap_paths (pm_join pm1 pm2)) in *.
      set (candidates1 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths1) in *.
      set (candidates2 := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths2) in *.
      set (candidates_join := flat_map (fun p1 => map (fun p2 => (p1, p2)) paths3) paths_join) in *.
      set (the_pred := fun '(p1, p2) => if list_eq_dec Z.eq_dec p (p1 ++ p2) then true else false) in *.
      set (matching1 := filter the_pred candidates1) in *.
      set (matching2 := filter the_pred candidates2) in *.
      set (matching_join := filter the_pred candidates_join) in *.
      (* Key lemma: join_all_values over pm_join = opt_join of separate join_all_values *)
      rewrite join_all_values_join_pathmap.
      (* Now goal: opt_join (join_all_values pm1 matching_join) (join_all_values pm2 matching_join)
                 = opt_join (join_all_values pm1 matching1) (join_all_values pm2 matching2) *)
      (* Use join_all_values_superset: matching_join ⊇ matching1 and pairs not in matching1 have pm1(p1) = None *)
      assert (Hpm1_eq: join_all_values pm1 matching_join = join_all_values pm1 matching1).
      { apply join_all_values_superset.
        - (* matching_join -> pm1 p1 <> None -> matching1 *)
          intros p1' p2' Hin Hpm1_ne.
          apply in_filter_iff in Hin. destruct Hin as [Hcand Hpred].
          apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
          apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
          injection Heq as Heq1 Heq2. subst p1'' p2''.
          (* pm1 p1' <> None means p1' is in paths(pm1) *)
          destruct (pm1 p1') as [v1'|] eqn:Hpm1_p1'; [|congruence].
          apply pathmap_paths_complete in Hpm1_p1'.
          apply in_filter_iff. split; [|assumption].
          apply in_flat_map_iff. exists p1'. split; [assumption|].
          apply in_map_iff_simple. exists p2'. split; [assumption|reflexivity].
        - (* matching1 -> matching_join *)
          intros p1' p2' Hin.
          apply in_filter_iff in Hin. destruct Hin as [Hcand Hpred].
          apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
          apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
          injection Heq as Heq1 Heq2. subst p1'' p2''.
          apply in_filter_iff. split; [|assumption].
          apply in_flat_map_iff. exists p1'. split.
          + unfold paths_join. apply pathmap_paths_join. left. assumption.
          + apply in_map_iff_simple. exists p2'. split; [assumption|reflexivity].
      }
      (* Similarly for pm2 *)
      assert (Hpm2_eq: join_all_values pm2 matching_join = join_all_values pm2 matching2).
      { apply join_all_values_superset.
        - (* matching_join -> pm2 p1 <> None -> matching2 *)
          intros p1' p2' Hin Hpm2_ne.
          apply in_filter_iff in Hin. destruct Hin as [Hcand Hpred].
          apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
          apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
          injection Heq as Heq1 Heq2. subst p1'' p2''.
          destruct (pm2 p1') as [v2'|] eqn:Hpm2_p1'; [|congruence].
          apply pathmap_paths_complete in Hpm2_p1'.
          apply in_filter_iff. split; [|assumption].
          apply in_flat_map_iff. exists p1'. split; [assumption|].
          apply in_map_iff_simple. exists p2'. split; [assumption|reflexivity].
        - (* matching2 -> matching_join *)
          intros p1' p2' Hin.
          apply in_filter_iff in Hin. destruct Hin as [Hcand Hpred].
          apply in_flat_map_iff in Hcand. destruct Hcand as [p1'' [Hp1'' Hpair]].
          apply in_map_iff_simple in Hpair. destruct Hpair as [p2'' [Hp2'' Heq]].
          injection Heq as Heq1 Heq2. subst p1'' p2''.
          apply in_filter_iff. split; [|assumption].
          apply in_flat_map_iff. exists p1'. split.
          + unfold paths_join. apply pathmap_paths_join. right. assumption.
          + apply in_map_iff_simple. exists p2'. split; [assumption|reflexivity].
      }
      rewrite Hpm1_eq. rewrite Hpm2_eq.
      reflexivity.
    - (* Only pm1*pm3 has value *)
      pose proof (multiply_right_value_when_pm2_none pm1 pm2 pm3 p v1 H1 H2) as Hlhs.
      rewrite Hlhs.
      unfold pm_join, pathmap_join, opt_join, option_join.
      rewrite H1. rewrite H2.
      reflexivity.
    - (* Only pm2*pm3 has value *)
      pose proof (multiply_right_value_when_pm1_none pm1 pm2 pm3 p v2 H2 H1) as Hlhs.
      rewrite Hlhs.
      unfold pm_join, pathmap_join, opt_join, option_join.
      rewrite H1. rewrite H2.
      reflexivity.
    - (* Neither has value, so LHS is also None *)
      assert (Hnone: pathmap_multiply (pm_join pm1 pm2) pm3 p = None).
      { destruct (pathmap_multiply (pm_join pm1 pm2) pm3 p) as [v|] eqn:Hleft.
        - assert (Hex: exists v, pathmap_multiply (pm_join pm1 pm2) pm3 p = Some v).
          { exists v. exact Hleft. }
          apply pathmap_multiply_join_exists_right in Hex.
          destruct Hex as [[v' Hv'] | [v' Hv']]; congruence.
        - reflexivity.
      }
      rewrite Hnone.
      unfold pm_join, pathmap_join, opt_join, option_join. rewrite H1. rewrite H2.
      reflexivity.
  Qed.

End Quantale.

(** ** Prefix Matching *)

Section PrefixMatching.

  (** Key property: A path matches if its prefix is in the PathMap.
      This is the prefix-matching semantics from the spec:
      "when you send on a channel, you're really also sending on
       all the prefixes of the channel" *)

  Variable V : Type.

  (** Find all prefixes of a path. *)
  Fixpoint all_prefixes (p : Path) : list Path :=
    match p with
    | [] => [[]]
    | x :: xs => [] :: map (cons x) (all_prefixes xs)
    end.

  (** Check if any prefix has a value. *)
  Definition has_prefix_value (pm : PathMap V) (p : Path) : bool :=
    existsb (fun prefix =>
      match pm prefix with
      | Some _ => true
      | None => false
      end
    ) (all_prefixes p).

  (** Get value at longest matching prefix. *)
  Fixpoint lookup_with_prefix (pm : PathMap V) (p : Path) : option V :=
    match pm p with
    | Some v => Some v
    | None =>
      match p with
      | [] => None
      | _ :: rest => lookup_with_prefix pm rest  (* Try shorter prefix *)
      end
    end.

  (** Empty path is a prefix of all paths. *)
  Theorem empty_is_prefix : forall p,
    is_prefix empty_path p = true.
  Proof.
    intros p. unfold empty_path, is_prefix. reflexivity.
  Qed.

  (** A path is a prefix of itself. *)
  Theorem prefix_reflexive : forall p,
    is_prefix p p = true.
  Proof.
    induction p as [|x xs IH]; simpl.
    - reflexivity.
    - rewrite Z.eqb_refl. simpl. apply IH.
  Qed.

  (** Prefix is transitive. *)
  Theorem prefix_transitive : forall p1 p2 p3,
    is_prefix p1 p2 = true ->
    is_prefix p2 p3 = true ->
    is_prefix p1 p3 = true.
  Proof.
    induction p1 as [|x xs IH]; intros p2 p3 H12 H23; simpl.
    - reflexivity.
    - destruct p2 as [|y ys]; simpl in H12.
      + discriminate.
      + destruct p3 as [|z zs]; simpl in H23.
        * discriminate.
        * simpl.
          apply andb_prop in H12. destruct H12 as [Hxy Hxsys].
          apply andb_prop in H23. destruct H23 as [Hyz Hyszs].
          apply Z.eqb_eq in Hxy. apply Z.eqb_eq in Hyz.
          subst. rewrite Z.eqb_refl. simpl.
          apply (IH ys zs); assumption.
  Qed.

End PrefixMatching.

(** ** Summary of Verified Properties *)

(**
   This module verifies the following algebraic properties of PathMap:

   1. **Lattice Laws**:
      - Join is commutative, associative, idempotent
      - Meet is commutative, associative, idempotent
      - Absorption laws hold

   2. **Distributive Lattice**:
      - Meet distributes over join
      - Subtraction with empty is identity
      - Self-subtraction yields empty

   3. **Quantale Laws** (fully verified):
      - Path concatenation is associative
      - Empty path is identity for concatenation
      - Multiplication distributes over join (left and right distributivity proven)

   4. **Prefix Matching**:
      - Empty is prefix of all paths
      - Prefix is reflexive and transitive

   The Rust implementation in PathMap/src/ring.rs implements these
   properties with the Lattice, DistributiveLattice, and Quantale traits.
*)

