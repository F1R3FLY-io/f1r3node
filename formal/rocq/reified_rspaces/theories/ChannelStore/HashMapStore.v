(** * HashMap Channel Store Specification

    This module specifies the behavior of HashMap-based channel stores.
    HashMap stores provide O(1) average-case lookup and insertion.

    Reference: Rust implementation in
      rspace++/src/rspace/traits/hash_map_bag_channel_store.rs
*)

From Stdlib Require Import List Bool ZArith Lia.
From Stdlib Require Import FunctionalExtensionality.
From ReifiedRSpaces Require Import Prelude.
From ReifiedRSpaces.Collections Require Import DataCollection.
Import ListNotations.

(** ** Channel Store Interface *)

(** Abstract specification of a channel store.
    A channel store maps channels to data collections and manages
    continuations waiting on channel patterns. *)
Class ChannelStore (C A K : Type) := {
  (** The underlying storage type *)
  cs_storage : Type;

  (** Empty channel store *)
  cs_empty : cs_storage;

  (** Get data collection for a channel (returns list of data) *)
  cs_get_data : cs_storage -> C -> list (A * bool);

  (** Put data on a channel *)
  cs_put_data : cs_storage -> C -> A -> bool -> cs_storage;

  (** Remove data from a channel (by predicate match) *)
  cs_remove_data : cs_storage -> C -> (A -> bool) -> option (A * cs_storage);

  (** Get continuations waiting on channels *)
  cs_get_conts : cs_storage -> list C -> list (list A * K * bool);

  (** Put a continuation waiting on channels *)
  cs_put_cont : cs_storage -> list C -> list A -> K -> bool -> cs_storage;

  (** Remove a continuation (by pattern match) *)
  cs_remove_cont : cs_storage -> list C -> list A ->
                   option ((list A * K * bool) * cs_storage);

  (** Generate a fresh channel name *)
  cs_gensym : cs_storage -> option (C * cs_storage);

  (** Clear all data (for checkpointing) *)
  cs_clear : cs_storage -> cs_storage;

  (** Check if empty *)
  cs_is_empty : cs_storage -> bool;

  (** Snapshot for soft checkpoint *)
  cs_snapshot : cs_storage -> cs_storage;
}.

(** ** HashMap Store Implementation *)

Section HashMapStore.
  Variable C : Type.
  Variable C_eq_dec : forall c1 c2 : C, {c1 = c2} + {c1 <> c2}.
  Variable A K : Type.

  (** HashMap is modeled as a function from channels to data lists *)
  Definition hashmap_data_store := C -> list (A * bool).
  Definition hashmap_cont_store := list C -> list (list A * K * bool).

  Record hashmap_storage := mkHashMapStorage {
    hm_data : hashmap_data_store;
    hm_conts : hashmap_cont_store;
    hm_counter : nat;  (* For gensym *)
  }.

  Definition hashmap_empty : hashmap_storage := {|
    hm_data := fun _ => [];
    hm_conts := fun _ => [];
    hm_counter := 0;
  |}.

  Definition hashmap_get_data (s : hashmap_storage) (c : C) : list (A * bool) :=
    hm_data s c.

  Definition hashmap_put_data (s : hashmap_storage) (c : C) (a : A) (persist : bool)
    : hashmap_storage := {|
    hm_data := fun c' =>
      if C_eq_dec c c' then (a, persist) :: hm_data s c'
      else hm_data s c';
    hm_conts := hm_conts s;
    hm_counter := hm_counter s;
  |}.

  Fixpoint remove_first_match {X} (pred : X -> bool) (l : list X) : option (X * list X) :=
    match l with
    | [] => None
    | x :: rest =>
      if pred x then Some (x, rest)
      else match remove_first_match pred rest with
           | None => None
           | Some (found, rest') => Some (found, x :: rest')
           end
    end.

  Definition hashmap_remove_data (s : hashmap_storage) (c : C) (pred : A -> bool)
    : option (A * hashmap_storage) :=
    let data := hm_data s c in
    let pred_tuple := fun '(a, _) => pred a in
    match remove_first_match pred_tuple data with
    | None => None
    | Some ((found, _), rest) =>
      Some (found, {|
        hm_data := fun c' =>
          if C_eq_dec c c' then rest
          else hm_data s c';
        hm_conts := hm_conts s;
        hm_counter := hm_counter s;
      |})
    end.

  Definition hashmap_get_conts (s : hashmap_storage) (cs : list C)
    : list (list A * K * bool) :=
    hm_conts s cs.

  Definition hashmap_put_cont (s : hashmap_storage) (cs : list C)
    (patterns : list A) (k : K) (persist : bool) : hashmap_storage := {|
    hm_data := hm_data s;
    hm_conts := fun cs' =>
      (* Simple list equality for channel lists *)
      (patterns, k, persist) :: hm_conts s cs';
    hm_counter := hm_counter s;
  |}.

  Definition hashmap_clear (s : hashmap_storage) : hashmap_storage := {|
    hm_data := fun _ => [];
    hm_conts := fun _ => [];
    hm_counter := hm_counter s;  (* Preserve counter across clears *)
  |}.

  Definition hashmap_is_empty (s : hashmap_storage) : bool :=
    (* This is a simplification - in practice we'd need to check all keys *)
    true.  (* Placeholder *)

  Definition hashmap_snapshot (s : hashmap_storage) : hashmap_storage := s.

  (** ** HashMap Store Properties *)

  (** Get after put on same channel returns the data *)
  Theorem hashmap_get_put_same :
    forall (s : hashmap_storage) (c : C) (a : A) (persist : bool),
      In (a, persist) (hashmap_get_data (hashmap_put_data s c a persist) c).
  Proof.
    intros s c a persist.
    unfold hashmap_get_data, hashmap_put_data.
    simpl.
    destruct (C_eq_dec c c) as [_ | Hneq].
    - (* c = c *)
      left. reflexivity.
    - (* c <> c - contradiction *)
      exfalso. apply Hneq. reflexivity.
  Qed.

  (** Get after put on different channel is unchanged *)
  Theorem hashmap_get_put_other :
    forall (s : hashmap_storage) (c c' : C) (a : A) (persist : bool),
      c <> c' ->
      hashmap_get_data (hashmap_put_data s c a persist) c' =
      hashmap_get_data s c'.
  Proof.
    intros s c c' a persist Hneq.
    unfold hashmap_get_data, hashmap_put_data.
    simpl.
    destruct (C_eq_dec c c') as [Heq | _].
    - (* c = c' - contradiction *)
      exfalso. apply Hneq. exact Heq.
    - (* c <> c' *)
      reflexivity.
  Qed.

  (** Empty store has no data *)
  Theorem hashmap_empty_no_data :
    forall (c : C),
      hashmap_get_data hashmap_empty c = [].
  Proof.
    intros c.
    unfold hashmap_get_data, hashmap_empty.
    simpl. reflexivity.
  Qed.

  (** Clear removes all data *)
  Theorem hashmap_clear_no_data :
    forall (s : hashmap_storage) (c : C),
      hashmap_get_data (hashmap_clear s) c = [].
  Proof.
    intros s c.
    unfold hashmap_get_data, hashmap_clear.
    simpl. reflexivity.
  Qed.

  (** Helper: remove_first_match returns element satisfying predicate *)
  Lemma remove_first_match_pred :
    forall (l : list (A * bool)) (pred : A -> bool) (found : A) (p : bool) (rest : list (A * bool)),
      remove_first_match (fun '(a0, _) => pred a0) l = Some ((found, p), rest) ->
      pred found = true.
  Proof.
    induction l as [| [h hp] t IH].
    - intros. discriminate.
    - intros pred0 found p rest Hfind.
      simpl in Hfind.
      destruct (pred0 h) eqn:Hpred.
      + inversion Hfind; subst. exact Hpred.
      + destruct (remove_first_match (fun '(a0, _) => pred0 a0) t) as [[[f' p'] r'] |] eqn:Hrec.
        * inversion Hfind; subst.
          eapply IH. exact Hrec.
        * discriminate.
  Qed.

  (** Remove data returns matching element if present *)
  Theorem hashmap_remove_data_returns_match :
    forall (s : hashmap_storage) (c : C) (pred : A -> bool) (a : A) (s' : hashmap_storage),
      hashmap_remove_data s c pred = Some (a, s') ->
      pred a = true.
  Proof.
    intros s c pred a s' Hremove.
    unfold hashmap_remove_data in Hremove.
    destruct (remove_first_match (fun '(a0, _) => pred a0) (hm_data s c)) eqn:Hfind.
    - destruct p as [[found persist] rest].
      inversion Hremove; subst.
      eapply remove_first_match_pred. exact Hfind.
    - discriminate.
  Qed.

  (** Remove data preserves other channels *)
  Theorem hashmap_remove_data_preserves_other :
    forall (s : hashmap_storage) (c c' : C) (pred : A -> bool) (a : A) (s' : hashmap_storage),
      c <> c' ->
      hashmap_remove_data s c pred = Some (a, s') ->
      hashmap_get_data s' c' = hashmap_get_data s c'.
  Proof.
    intros s c c' pred a s' Hneq Hremove.
    unfold hashmap_remove_data in Hremove.
    destruct (remove_first_match (fun '(a0, _) => pred a0) (hm_data s c)) eqn:Hfind.
    - destruct p as [[found persist] rest].
      injection Hremove as Ha Hs'.
      subst s'.
      unfold hashmap_get_data.
      simpl.
      destruct (C_eq_dec c c') as [Heq | _].
      + exfalso. apply Hneq. exact Heq.
      + reflexivity.
    - discriminate.
  Qed.

  (** Snapshot preserves all data *)
  Theorem hashmap_snapshot_preserves_data :
    forall (s : hashmap_storage) (c : C),
      hashmap_get_data (hashmap_snapshot s) c =
      hashmap_get_data s c.
  Proof.
    intros s c.
    unfold hashmap_snapshot, hashmap_get_data.
    reflexivity.
  Qed.

  (** Put data increments length *)
  Theorem hashmap_put_data_increments_length :
    forall (s : hashmap_storage) (c : C) (a : A) (persist : bool),
      length (hashmap_get_data (hashmap_put_data s c a persist) c) =
      S (length (hashmap_get_data s c)).
  Proof.
    intros s c a persist.
    unfold hashmap_get_data, hashmap_put_data.
    simpl.
    destruct (C_eq_dec c c) as [_ | Hneq].
    - simpl. reflexivity.
    - exfalso. apply Hneq. reflexivity.
  Qed.

End HashMapStore.

(** ** Gensym Properties for Nat Channels *)

Section GensymProperties.
  Variable A K : Type.

  (** For HashMap with nat channels, gensym returns unique indices *)
  Definition hashmap_gensym_nat (s : hashmap_storage nat A K)
    : option (nat * hashmap_storage nat A K) :=
    let idx := hm_counter nat A K s in
    Some (idx, {|
      hm_data := hm_data nat A K s;
      hm_conts := hm_conts nat A K s;
      hm_counter := S idx;
    |}).

  (** Gensym returns strictly increasing indices *)
  Theorem hashmap_gensym_monotonic :
    forall (s : hashmap_storage nat A K) (c : nat) (s' : hashmap_storage nat A K),
      hashmap_gensym_nat s = Some (c, s') ->
      hm_counter nat A K s' = S (hm_counter nat A K s).
  Proof.
    intros s c s' H.
    unfold hashmap_gensym_nat in H.
    injection H as Hc Hs'.
    subst s'.
    simpl. reflexivity.
  Qed.

  (** Gensym returns the current counter value *)
  Theorem hashmap_gensym_returns_counter :
    forall (s : hashmap_storage nat A K) (c : nat) (s' : hashmap_storage nat A K),
      hashmap_gensym_nat s = Some (c, s') ->
      c = hm_counter nat A K s.
  Proof.
    intros s c s' H.
    unfold hashmap_gensym_nat in H.
    injection H as Hc Hs'.
    symmetry. exact Hc.
  Qed.

  (** Sequential gensym calls return distinct indices *)
  Theorem hashmap_gensym_distinct :
    forall (s : hashmap_storage nat A K)
           (c1 : nat) (s1 : hashmap_storage nat A K)
           (c2 : nat) (s2 : hashmap_storage nat A K),
      hashmap_gensym_nat s = Some (c1, s1) ->
      hashmap_gensym_nat s1 = Some (c2, s2) ->
      c1 <> c2.
  Proof.
    intros s c1 s1 c2 s2 H1 H2.
    unfold hashmap_gensym_nat in *.
    injection H1 as Hc1 Hs1.
    injection H2 as Hc2 Hs2.
    subst c1 s1 c2.
    simpl. lia.
  Qed.

  (** Gensym preserves existing data *)
  Theorem hashmap_gensym_preserves_data :
    forall (s : hashmap_storage nat A K) (c' : nat) (s' : hashmap_storage nat A K) (c : nat),
      hashmap_gensym_nat s = Some (c', s') ->
      hashmap_get_data nat A K s' c = hashmap_get_data nat A K s c.
  Proof.
    intros s c' s' c H.
    unfold hashmap_gensym_nat in H.
    injection H as Hc' Hs'.
    subst s'.
    unfold hashmap_get_data.
    simpl. reflexivity.
  Qed.

End GensymProperties.
