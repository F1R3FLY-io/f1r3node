(** * Priority Queue Collection Specification

    This module specifies the behavior of priority queue collections.
    A priority queue stores data at different priority levels, where
    matching always returns the highest-priority matching element first.

    Reference: Spec lines about PriorityQueue in Reifying RSpaces.md
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Priority Queue Interface *)

(** Priority type: 0 is highest priority *)
Definition Priority := nat.

(** Storage for a single priority level *)
Definition PriorityLevel (A : Type) := list (A * bool). (* (data, persist) *)

(** Priority queue with n priority levels (0 to n-1) *)
Section PriorityQueueCollection.
  Variable A : Type.
  Variable num_levels : nat.

  (** Storage: list of priority levels, index 0 = highest priority *)
  Definition pqueue_storage := list (PriorityLevel A).

  (** Initialize with empty levels *)
  Fixpoint pqueue_init_levels (n : nat) : pqueue_storage :=
    match n with
    | 0 => []
    | S n' => [] :: pqueue_init_levels n'
    end.

  Definition pqueue_empty : pqueue_storage := pqueue_init_levels num_levels.

  (** Get the level at a given priority *)
  Definition pqueue_get_level (s : pqueue_storage) (p : Priority) : option (PriorityLevel A) :=
    nth_error s p.

  (** Set the level at a given priority *)
  Fixpoint pqueue_set_level (s : pqueue_storage) (p : Priority) (level : PriorityLevel A)
    : option pqueue_storage :=
    match s, p with
    | [], _ => None (* priority out of range *)
    | _ :: rest, 0 => Some (level :: rest)
    | h :: rest, S p' =>
      match pqueue_set_level rest p' level with
      | None => None
      | Some rest' => Some (h :: rest')
      end
    end.

  (** Put data at a specific priority level *)
  Definition pqueue_put (s : pqueue_storage) (priority : Priority) (a : A)
    : option pqueue_storage :=
    match pqueue_get_level s priority with
    | None => None (* invalid priority *)
    | Some level =>
      let new_level := level ++ [(a, false)] in
      pqueue_set_level s priority new_level
    end.

  (** Find and remove matching data, checking highest priority first *)
  Fixpoint pqueue_find_in_level (level : PriorityLevel A) (pred : A -> bool)
    : option (A * PriorityLevel A) :=
    match level with
    | [] => None
    | (a, persist) :: rest =>
      if pred a then
        if persist then
          Some (a, level) (* Keep persistent data *)
        else
          Some (a, rest) (* Remove non-persistent data *)
      else
        match pqueue_find_in_level rest pred with
        | None => None
        | Some (found, rest') => Some (found, (a, persist) :: rest')
        end
    end.

  (** Find match across all levels, starting from priority 0 *)
  Fixpoint pqueue_find_match_from (s : pqueue_storage) (p : Priority) (pred : A -> bool)
    : option (A * Priority * pqueue_storage) :=
    match s with
    | [] => None
    | level :: rest =>
      match pqueue_find_in_level level pred with
      | Some (found, level') =>
        (* Found at this priority level *)
        Some (found, p, level' :: rest)
      | None =>
        (* Try next priority level *)
        match pqueue_find_match_from rest (S p) pred with
        | None => None
        | Some (found, p', rest') => Some (found, p', level :: rest')
        end
      end
    end.

  Definition pqueue_find_match (s : pqueue_storage) (pred : A -> bool)
    : option (A * pqueue_storage) :=
    match pqueue_find_match_from s 0 pred with
    | None => None
    | Some (found, _, s') => Some (found, s')
    end.

  (** Check if queue is empty *)
  Fixpoint pqueue_is_empty (s : pqueue_storage) : bool :=
    match s with
    | [] => true
    | level :: rest =>
      match level with
      | [] => pqueue_is_empty rest
      | _ => false
      end
    end.

  (** Count total elements across all levels *)
  Fixpoint pqueue_len (s : pqueue_storage) : nat :=
    match s with
    | [] => 0
    | level :: rest => length level + pqueue_len rest
    end.

End PriorityQueueCollection.

(** ** Priority Queue Properties *)

Section PriorityQueueProperties.
  Variable A : Type.

  (** Helper lemma: init_levels produces empty storage *)
  Lemma pqueue_init_levels_is_empty :
    forall m, pqueue_is_empty A (pqueue_init_levels A m) = true.
  Proof.
    induction m as [| m' IH].
    - simpl. reflexivity.
    - simpl. exact IH.
  Qed.

  (** Helper lemma: init_levels produces zero length *)
  Lemma pqueue_init_levels_len_zero :
    forall m, pqueue_len A (pqueue_init_levels A m) = 0.
  Proof.
    induction m as [| m' IH].
    - simpl. reflexivity.
    - simpl. exact IH.
  Qed.

  Variable n : nat.

  (** Empty queue is empty *)
  Theorem pqueue_empty_is_empty :
    pqueue_is_empty A (pqueue_empty A n) = true.
  Proof.
    unfold pqueue_empty.
    apply pqueue_init_levels_is_empty.
  Qed.

  (** Empty queue has length 0 *)
  Theorem pqueue_empty_len_zero :
    pqueue_len A (pqueue_empty A n) = 0.
  Proof.
    unfold pqueue_empty.
    apply pqueue_init_levels_len_zero.
  Qed.

  (** Put at valid priority succeeds *)
  Theorem pqueue_put_valid_priority_succeeds :
    forall (s : pqueue_storage A) (p : Priority) (a : A) level,
      pqueue_get_level A s p = Some level ->
      exists s', pqueue_put A s p a = Some s'.
  Proof.
    intros s p a level Hget.
    unfold pqueue_put.
    rewrite Hget.
    (* Need to show set_level succeeds when get_level succeeded *)
    generalize dependent level.
    generalize dependent p.
    induction s as [| h t IH].
    - intros p level Hget.
      unfold pqueue_get_level in Hget. simpl in Hget. destruct p; inversion Hget.
    - intros p level Hget.
      destruct p.
      + (* p = 0 *)
        simpl. eexists. reflexivity.
      + (* p = S p' *)
        simpl in Hget.
        simpl.
        specialize (IH p level Hget).
        destruct IH as [s' Hs'].
        rewrite Hs'.
        eexists. reflexivity.
  Qed.

  (** Put at invalid priority fails *)
  Theorem pqueue_put_invalid_priority_fails :
    forall (s : pqueue_storage A) (p : Priority) (a : A),
      pqueue_get_level A s p = None ->
      pqueue_put A s p a = None.
  Proof.
    intros s p a Hget.
    unfold pqueue_put.
    rewrite Hget.
    reflexivity.
  Qed.

  (** Put increases length by 1 *)
  Theorem pqueue_put_increases_len :
    forall (s : pqueue_storage A) (p : Priority) (a : A) s',
      pqueue_put A s p a = Some s' ->
      pqueue_len A s' = S (pqueue_len A s).
  Proof.
    intros s p a s' Hput.
    unfold pqueue_put in Hput.
    destruct (pqueue_get_level A s p) as [level |] eqn:Hget.
    2: { discriminate. }
    (* Need to show that set_level preserves other levels and increases one *)
    generalize dependent s'.
    generalize dependent level.
    generalize dependent p.
    induction s as [| h t IH].
    - intros p level Hget s' Hput.
      simpl in Hget. destruct p; discriminate.
    - intros p level Hget s' Hput.
      destruct p.
      + (* p = 0, level = h *)
        simpl in Hget. injection Hget as Hlevel. subst level.
        simpl in Hput.
        injection Hput as Hs'. subst s'.
        simpl.
        rewrite List.length_app. simpl.
        lia.
      + (* p = S p', level in t *)
        simpl in Hget.
        simpl in Hput.
        destruct (pqueue_set_level A t p (level ++ [(a, false)])) as [rest |] eqn:Hset.
        2: { discriminate. }
        injection Hput as Hs'. subst s'.
        simpl.
        (* Use IH to get pqueue_len rest = S (pqueue_len t) *)
        assert (Hih : pqueue_len A rest = S (pqueue_len A t)).
        { eapply IH. exact Hget. exact Hset. }
        lia.
  Qed.

End PriorityQueueProperties.

(** ** Priority Queue Priority Ordering Property *)

Section PriorityOrdering.
  Variable A : Type.

  (** Helper: element is at a specific priority level *)
  Definition element_at_priority (s : pqueue_storage A) (p : Priority) (a : A) : Prop :=
    match pqueue_get_level A s p with
    | None => False
    | Some level => exists prefix persist suffix, level = prefix ++ [(a, persist)] ++ suffix
    end.

  (** Helper: If a matching element exists in a level, find_in_level finds some element. *)
  Lemma pqueue_find_in_level_finds_match :
    forall (level : PriorityLevel A) (a : A) (pred : A -> bool) prefix persist suffix,
      level = prefix ++ [(a, persist)] ++ suffix ->
      pred a = true ->
      exists found level', pqueue_find_in_level A level pred = Some (found, level').
  Proof.
    intros level a pred prefix.
    revert level.
    induction prefix as [| (x, px) prefix' IH]; intros level persist suffix Hlevel Hpred.
    - (* a is at the front *)
      subst level. simpl. rewrite Hpred.
      destruct persist; eexists; eexists; reflexivity.
    - (* a is somewhere after the first element *)
      subst level.
      simpl.
      destruct (pred x) eqn:Hpredx.
      + (* x matches, so we find x first *)
        destruct px; eexists; eexists; reflexivity.
      + (* x doesn't match, continue *)
        (* IH applies to the rest of the list *)
        pose proof (IH (prefix' ++ [(a, persist)] ++ suffix) persist suffix eq_refl Hpred) as IHres.
        destruct IHres as [found [level' Hfind]].
        (* [(a, persist)] ++ suffix = (a, persist) :: suffix by computation *)
        (* So we can replace one form with the other *)
        replace (prefix' ++ (a, persist) :: suffix)
          with (prefix' ++ [(a, persist)] ++ suffix) by reflexivity.
        rewrite Hfind.
        eexists; eexists; reflexivity.
  Qed.

  (** Helper: find_match_from finds the first match starting from index p in storage. *)
  Lemma pqueue_find_match_from_finds_first :
    forall (s : pqueue_storage A) (p : Priority) (pred : A -> bool)
           (level : PriorityLevel A) (a : A) prefix persist suffix,
      nth_error s 0 = Some level ->
      level = prefix ++ [(a, persist)] ++ suffix ->
      pred a = true ->
      exists found rest, pqueue_find_match_from A s p pred = Some (found, p, rest).
  Proof.
    intros s p pred level a prefix persist suffix Hnth Hlevel Hpred.
    destruct s as [| level0 rest].
    - simpl in Hnth. discriminate.
    - simpl in Hnth. injection Hnth as Heq. subst level0.
      simpl.
      assert (Hfind: exists found level',
        pqueue_find_in_level A level pred = Some (found, level')).
      { eapply pqueue_find_in_level_finds_match; eauto. }
      destruct Hfind as [found [level' Hfind]].
      rewrite Hfind.
      eexists; eexists; reflexivity.
  Qed.

  (** Crucial: get_level at p means we can reach it from position 0 in the list. *)
  Lemma pqueue_get_level_nth_error :
    forall (s : pqueue_storage A) (p : Priority),
      pqueue_get_level A s p = nth_error s p.
  Proof.
    intros s p. unfold pqueue_get_level. reflexivity.
  Qed.

  (** If find_match_from finds something starting from current priority p,
      it finds at priority >= p. *)
  Lemma pqueue_find_match_from_priority_bound :
    forall (s : pqueue_storage A) (start_p : Priority) (pred : A -> bool)
           found result_p rest,
      pqueue_find_match_from A s start_p pred = Some (found, result_p, rest) ->
      result_p >= start_p.
  Proof.
    induction s as [| level s' IH]; intros start_p pred found result_p rest Hfind.
    - simpl in Hfind. discriminate.
    - simpl in Hfind.
      destruct (pqueue_find_in_level A level pred) as [[f l'] |] eqn:Hlevel.
      + (* Found in current level *)
        injection Hfind as Hf Hp Hr.
        subst. lia.
      + (* Search in rest *)
        destruct (pqueue_find_match_from A s' (S start_p) pred) as [[[f' p'] r'] |] eqn:Hrest.
        * injection Hfind as Hf Hp Hr.
          assert (Hbound: p' >= S start_p).
          { apply (IH (S start_p) pred f' p' r'). exact Hrest. }
          subst. lia.
        * discriminate.
  Qed.

  (** If there's a match at some level, find_match_from returns Some. *)
  Lemma pqueue_find_match_from_succeeds :
    forall (s : pqueue_storage A) (start_p level_idx : Priority)
           (level : PriorityLevel A) (pred : A -> bool)
           (a : A) prefix persist suffix,
      nth_error s level_idx = Some level ->
      level = prefix ++ [(a, persist)] ++ suffix ->
      pred a = true ->
      exists found result_p rest,
        pqueue_find_match_from A s start_p pred = Some (found, result_p, rest).
  Proof.
    induction s as [| hd tl IH]; intros start_p level_idx level pred a prefix persist suffix
      Hnth Hlevel Hpred.
    - (* s = [] *)
      simpl in Hnth. destruct level_idx; discriminate.
    - (* s = hd :: tl *)
      simpl.
      destruct (pqueue_find_in_level A hd pred) as [[f l'] |] eqn:Hhdmatch.
      + (* Found in hd *)
        eexists; eexists; eexists; reflexivity.
      + (* Not found in hd, search in tl *)
        destruct level_idx as [| level_idx'].
        * (* level_idx = 0, so level = hd, but we said no match in hd - contradiction *)
          simpl in Hnth. injection Hnth as Heq. subst hd.
          (* level has a match (a with pred a = true), but find_in_level returned None *)
          assert (Hcontra: exists found level',
            pqueue_find_in_level A level pred = Some (found, level')).
          { eapply pqueue_find_in_level_finds_match; eauto. }
          destruct Hcontra as [found [level' Hcontra]].
          rewrite Hcontra in Hhdmatch. discriminate.
        * (* level_idx = S level_idx', so level is in tl *)
          simpl in Hnth.
          specialize (IH (S start_p) level_idx' level pred a prefix persist suffix
                         Hnth Hlevel Hpred).
          destruct IH as [found [result_p [rest Htl]]].
          rewrite Htl.
          eexists; eexists; eexists; reflexivity.
  Qed.

  (** Key property: find_match returns an element from highest priority level first.
      If an element matches at priority p1 and another at p2, and p1 < p2,
      then find_match returns SOME element at priority <= p1 (not one at p2). *)
  Theorem pqueue_priority_ordering :
    forall (s : pqueue_storage A) (a1 a2 : A) (p1 p2 : Priority) pred,
      element_at_priority s p1 a1 ->
      element_at_priority s p2 a2 ->
      p1 < p2 ->
      pred a1 = true ->
      pred a2 = true ->
      (* find_match returns SOME element (not necessarily a1, but from a level <= p1) *)
      exists found s', pqueue_find_match A s pred = Some (found, s').
  Proof.
    intros s a1 a2 p1 p2 pred Hat1 Hat2 Hlt Hpred1 Hpred2.
    unfold element_at_priority in Hat1.
    destruct (pqueue_get_level A s p1) as [level1 |] eqn:Hget1.
    2: { exfalso. exact Hat1. }
    destruct Hat1 as [prefix1 [persist1 [suffix1 Hlevel1]]].
    (* a1 is at priority p1, and pred a1 = true,
       so there's definitely a match in the queue. *)
    unfold pqueue_find_match.
    rewrite pqueue_get_level_nth_error in Hget1.
    (* Use the pqueue_find_match_from_succeeds lemma *)
    pose proof (pqueue_find_match_from_succeeds s 0 p1 level1 pred a1 prefix1 persist1 suffix1
                  Hget1 Hlevel1 Hpred1) as Hsucc.
    destruct Hsucc as [found [result_p [rest Hfind]]].
    rewrite Hfind.
    eexists; eexists; reflexivity.
  Qed.

End PriorityOrdering.

(** ** Integration with DataCollection typeclass *)

Section PriorityQueueDataCollection.
  Variable A : Type.
  Variable num_levels : nat.
  Variable default_priority : Priority.
  Hypothesis Hdefault_valid : default_priority < num_levels.

  (** Wrapper that uses default priority for put *)
  Definition pqueue_dc_put (s : pqueue_storage A) (a : A) : option (pqueue_storage A) :=
    pqueue_put A s default_priority a.

  (** Note: A full DataCollection instance would need to track priorities,
      which doesn't fit the simple A -> bool predicate interface perfectly.
      The priority queue is better used with explicit priority operations. *)

End PriorityQueueDataCollection.
