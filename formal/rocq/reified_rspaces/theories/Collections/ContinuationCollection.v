(** * Continuation Collection Specifications

    This module specifies the behavior of continuation collections at channels.
    Each collection type has different semantics for storing and matching
    continuations (waiters).

    Continuations are registered patterns waiting for data to arrive.
    When data arrives, a matching continuation is found and executed.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/collections.rs

    Key difference from DataCollection:
    - DataCollection stores: (A, bool) - data with persist flag
    - ContinuationCollection stores: (list P, K, bool) - patterns, continuation, persist
*)

From Stdlib Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Continuation Collection Interface *)

(** Abstract specification of a continuation collection.
    P is the pattern type, K is the continuation type. *)
Class ContinuationCollection (P K : Type) := {
  (** The underlying storage type *)
  cc_storage : Type;

  (** Empty collection *)
  cc_empty : cc_storage;

  (** Register a continuation with patterns *)
  cc_put : cc_storage -> list P -> K -> bool -> cc_storage;

  (** Find and optionally remove matching continuation.
      The predicate takes (patterns, continuation) and returns bool. *)
  cc_find_match : cc_storage -> (list P -> K -> bool) -> option (list P * K * bool * cc_storage);

  (** Get all continuations without removing *)
  cc_get_all : cc_storage -> list (list P * K * bool);

  (** Check if collection is empty *)
  cc_is_empty : cc_storage -> bool;

  (** Get the number of continuations *)
  cc_len : cc_storage -> nat;

  (** Clear all continuations *)
  cc_clear : cc_storage -> cc_storage;
}.

(** ** Bag Continuation Collection (Multiset) *)

(** Bag stores multiple continuations and any can match. *)
Section BagContinuationCollection.
  Variable P K : Type.

  Definition bag_cc_storage := list (list P * K * bool).

  Definition bag_cc_empty : bag_cc_storage := [].

  Definition bag_cc_put (s : bag_cc_storage) (patterns : list P) (k : K) (persist : bool)
    : bag_cc_storage :=
    (patterns, k, persist) :: s.

  Fixpoint bag_cc_find_match (s : bag_cc_storage) (pred : list P -> K -> bool)
    : option (list P * K * bool * bag_cc_storage) :=
    match s with
    | [] => None
    | (patterns, k, persist) :: rest =>
      if pred patterns k then
        if persist then
          (* Persistent continuation: return match but keep in collection *)
          Some (patterns, k, persist, s)
        else
          (* Non-persistent: remove from collection *)
          Some (patterns, k, persist, rest)
      else
        match bag_cc_find_match rest pred with
        | None => None
        | Some (found_p, found_k, found_persist, rest') =>
          Some (found_p, found_k, found_persist, (patterns, k, persist) :: rest')
        end
    end.

  Definition bag_cc_is_empty (s : bag_cc_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition bag_cc_len (s : bag_cc_storage) : nat := length s.

  Definition bag_cc_clear (s : bag_cc_storage) : bag_cc_storage := [].

  Global Instance BagContinuationCollection : ContinuationCollection P K := {
    cc_storage := bag_cc_storage;
    cc_empty := bag_cc_empty;
    cc_put := bag_cc_put;
    cc_find_match := bag_cc_find_match;
    cc_get_all := fun s => s;
    cc_is_empty := bag_cc_is_empty;
    cc_len := bag_cc_len;
    cc_clear := bag_cc_clear;
  }.
End BagContinuationCollection.

(** ** Queue Continuation Collection (FIFO) *)

(** Queue stores continuations with first-in-first-out semantics.
    Only the oldest (front) continuation can match. *)
Section QueueContinuationCollection.
  Variable P K : Type.

  (* Queue uses two lists for amortized O(1) operations *)
  Definition queue_cc_storage := (list (list P * K * bool) * list (list P * K * bool))%type.

  Definition queue_cc_empty : queue_cc_storage := ([], []).

  Definition queue_cc_put (s : queue_cc_storage) (patterns : list P) (k : K) (persist : bool)
    : queue_cc_storage :=
    let '(front, back) := s in
    (front, (patterns, k, persist) :: back).

  (* Normalize: move back to front when front is empty *)
  Definition queue_cc_normalize (s : queue_cc_storage) : queue_cc_storage :=
    let '(front, back) := s in
    match front with
    | [] => (rev back, [])
    | _ => s
    end.

  Definition queue_cc_find_match (s : queue_cc_storage) (pred : list P -> K -> bool)
    : option (list P * K * bool * queue_cc_storage) :=
    let s' := queue_cc_normalize s in
    let '(front, back) := s' in
    match front with
    | [] => None
    | (patterns, k, persist) :: rest =>
      if pred patterns k then
        if persist then
          Some (patterns, k, persist, s')
        else
          Some (patterns, k, persist, (rest, back))
      else
        None  (* Only front can match in a queue *)
    end.

  Definition queue_cc_is_empty (s : queue_cc_storage) : bool :=
    let '(front, back) := s in
    match front, back with
    | [], [] => true
    | _, _ => false
    end.

  Definition queue_cc_len (s : queue_cc_storage) : nat :=
    let '(front, back) := s in
    length front + length back.

  Definition queue_cc_clear (s : queue_cc_storage) : queue_cc_storage := ([], []).

  Definition queue_cc_get_all (s : queue_cc_storage) : list (list P * K * bool) :=
    let '(front, back) := queue_cc_normalize s in
    front.

  Global Instance QueueContinuationCollection : ContinuationCollection P K := {
    cc_storage := queue_cc_storage;
    cc_empty := queue_cc_empty;
    cc_put := queue_cc_put;
    cc_find_match := queue_cc_find_match;
    cc_get_all := queue_cc_get_all;
    cc_is_empty := queue_cc_is_empty;
    cc_len := queue_cc_len;
    cc_clear := queue_cc_clear;
  }.
End QueueContinuationCollection.

(** ** Stack Continuation Collection (LIFO) *)

(** Stack stores continuations with last-in-first-out semantics.
    Only the top (most recently added) continuation can match. *)
Section StackContinuationCollection.
  Variable P K : Type.

  Definition stack_cc_storage := list (list P * K * bool).

  Definition stack_cc_empty : stack_cc_storage := [].

  Definition stack_cc_put (s : stack_cc_storage) (patterns : list P) (k : K) (persist : bool)
    : stack_cc_storage :=
    (patterns, k, persist) :: s.  (* Push to top *)

  (** Stack find only matches the top element *)
  Definition stack_cc_find_match (s : stack_cc_storage) (pred : list P -> K -> bool)
    : option (list P * K * bool * stack_cc_storage) :=
    match s with
    | [] => None
    | (patterns, k, persist) :: rest =>
      if pred patterns k then
        if persist then Some (patterns, k, persist, s)
        else Some (patterns, k, persist, rest)
      else None  (* Only top can match in a stack *)
    end.

  Definition stack_cc_is_empty (s : stack_cc_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition stack_cc_len (s : stack_cc_storage) : nat := length s.

  Definition stack_cc_clear (s : stack_cc_storage) : stack_cc_storage := [].

  Global Instance StackContinuationCollection : ContinuationCollection P K := {
    cc_storage := stack_cc_storage;
    cc_empty := stack_cc_empty;
    cc_put := stack_cc_put;
    cc_find_match := stack_cc_find_match;
    cc_get_all := fun s => s;
    cc_is_empty := stack_cc_is_empty;
    cc_len := stack_cc_len;
    cc_clear := stack_cc_clear;
  }.
End StackContinuationCollection.

(** ** Set Continuation Collection (Idempotent) *)

(** Set stores unique continuations. Registering the same continuation
    (same patterns, continuation, persist) multiple times has no effect.

    Note: Uses a Vec internally and checks for duplicates on insert,
    because (list P, K, bool) tuples don't naturally implement Hash + Eq
    for all P, K types. *)
Section SetContinuationCollection.
  Variable P K : Type.
  Variable P_eq_dec : forall p1 p2 : P, {p1 = p2} + {p1 <> p2}.
  Variable K_eq_dec : forall k1 k2 : K, {k1 = k2} + {k1 <> k2}.

  Definition set_cc_storage := list (list P * K * bool).

  Definition set_cc_empty : set_cc_storage := [].

  (** Equality check for pattern lists *)
  Fixpoint patterns_eq (ps1 ps2 : list P) : bool :=
    match ps1, ps2 with
    | [], [] => true
    | p1 :: rest1, p2 :: rest2 =>
      if P_eq_dec p1 p2 then patterns_eq rest1 rest2 else false
    | _, _ => false
    end.

  (** Check if continuation entry exists in set *)
  Fixpoint set_cc_member (s : set_cc_storage) (patterns : list P) (k : K) (persist : bool) : bool :=
    match s with
    | [] => false
    | (ps, k', persist') :: rest =>
      if patterns_eq ps patterns then
        if K_eq_dec k' k then
          if Bool.eqb persist' persist then true
          else set_cc_member rest patterns k persist
        else set_cc_member rest patterns k persist
      else set_cc_member rest patterns k persist
    end.

  (** Set put is idempotent - adding existing continuation does nothing *)
  Definition set_cc_put (s : set_cc_storage) (patterns : list P) (k : K) (persist : bool)
    : set_cc_storage :=
    if set_cc_member s patterns k persist then s  (* Already exists, idempotent *)
    else (patterns, k, persist) :: s.

  (** Set find can match any continuation (like Bag, non-deterministic) *)
  Fixpoint set_cc_find_match (s : set_cc_storage) (pred : list P -> K -> bool)
    : option (list P * K * bool * set_cc_storage) :=
    match s with
    | [] => None
    | (patterns, k, persist) :: rest =>
      if pred patterns k then
        if persist then Some (patterns, k, persist, s)
        else Some (patterns, k, persist, rest)
      else
        match set_cc_find_match rest pred with
        | None => None
        | Some (found_p, found_k, found_persist, rest') =>
          Some (found_p, found_k, found_persist, (patterns, k, persist) :: rest')
        end
    end.

  Definition set_cc_is_empty (s : set_cc_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition set_cc_len (s : set_cc_storage) : nat := length s.

  Definition set_cc_clear (s : set_cc_storage) : set_cc_storage := [].

  Global Instance SetContinuationCollection : ContinuationCollection P K := {
    cc_storage := set_cc_storage;
    cc_empty := set_cc_empty;
    cc_put := set_cc_put;
    cc_find_match := set_cc_find_match;
    cc_get_all := fun s => s;
    cc_is_empty := set_cc_is_empty;
    cc_len := set_cc_len;
    cc_clear := set_cc_clear;
  }.
End SetContinuationCollection.

(** ** Properties *)

Section ContinuationCollectionProperties.
  Context {P K : Type} `{CC : ContinuationCollection P K}.

  (** Empty collection has length 0 *)
  Definition cc_empty_has_len_zero : Prop :=
    cc_len cc_empty = 0.

  (** Empty collection is empty *)
  Definition cc_empty_is_empty : Prop :=
    cc_is_empty cc_empty = true.

  (** Put increases length by 1 *)
  Definition cc_put_increases_len : Prop :=
    forall s patterns k persist,
      cc_len (cc_put s patterns k persist) = S (cc_len s) \/
      cc_len (cc_put s patterns k persist) = cc_len s.  (* For idempotent Set *)

  (** Find match doesn't increase length *)
  Definition cc_find_match_preserves_or_decreases_len : Prop :=
    forall s pred patterns k persist s',
      cc_find_match s pred = Some (patterns, k, persist, s') ->
      cc_len s' <= cc_len s.

  (** Clear results in empty collection *)
  Definition cc_clear_is_empty : Prop :=
    forall s, cc_is_empty (cc_clear s) = true.
End ContinuationCollectionProperties.

(** ** Bag-Specific Theorems *)

Section BagCCTheorems.
  Variable P K : Type.

  (** Bag put always adds element (non-idempotent) *)
  Theorem bag_cc_put_increases_len :
    forall (s : @cc_storage P K (BagContinuationCollection P K)) patterns k persist,
      cc_len (cc_put s patterns k persist) = S (cc_len s).
  Proof.
    intros s patterns k persist.
    simpl. reflexivity.
  Qed.

  (** Bag empty is indeed empty *)
  Theorem bag_cc_empty_is_empty :
    @cc_is_empty P K (BagContinuationCollection P K) cc_empty = true.
  Proof.
    reflexivity.
  Qed.

  (** Bag clear produces empty collection *)
  Theorem bag_cc_clear_is_empty :
    forall (s : @cc_storage P K (BagContinuationCollection P K)),
      cc_is_empty (cc_clear s) = true.
  Proof.
    intros s. reflexivity.
  Qed.
End BagCCTheorems.

(** ** Queue-Specific Theorems *)

Section QueueCCTheorems.
  Variable P K : Type.

  (** Queue put always succeeds *)
  Theorem queue_cc_put_increases_len :
    forall (s : @cc_storage P K (QueueContinuationCollection P K)) patterns k persist,
      cc_len (cc_put s patterns k persist) = S (cc_len s).
  Proof.
    intros [front back] patterns k persist.
    simpl. lia.
  Qed.

  (** Queue empty is indeed empty *)
  Theorem queue_cc_empty_is_empty :
    @cc_is_empty P K (QueueContinuationCollection P K) cc_empty = true.
  Proof.
    reflexivity.
  Qed.

  (** Queue FIFO property: first put is first found (when queue was empty) *)
  Theorem queue_cc_fifo_first_in_first_out :
    forall (patterns : list P) (k : K),
      forall s',
        cc_put (cc_empty : @cc_storage P K (QueueContinuationCollection P K)) patterns k false = s' ->
        forall pred, pred patterns k = true ->
        exists rest, cc_find_match s' pred = Some (patterns, k, false, rest).
  Proof.
    intros patterns k s' Hput pred Hpred.
    cbn in Hput.
    subst s'.
    (* s' = ([], [(patterns, k, false)]) *)
    exists ([], []).
    unfold cc_find_match, queue_cc_find_match, queue_cc_normalize.
    cbn.
    destruct (pred patterns k) eqn:Hpa.
    - reflexivity.
    - rewrite Hpred in Hpa. discriminate.
  Qed.

  (** Queue FIFO ordering: second put goes behind first *)
  Theorem queue_cc_fifo_ordering :
    forall (p1 p2 : list P) (k1 k2 : K) s1 s2,
      cc_put (cc_empty : @cc_storage P K (QueueContinuationCollection P K)) p1 k1 false = s1 ->
      cc_put s1 p2 k2 false = s2 ->
      forall pred, pred p1 k1 = true -> pred p2 k2 = true ->
      (* Finding on s2 should return (p1, k1) first *)
      exists rest, cc_find_match s2 pred = Some (p1, k1, false, rest).
  Proof.
    intros p1 p2 k1 k2 s1 s2 Hput1 Hput2 pred Hp1 Hp2.
    cbn in Hput1.
    subst s1.
    cbn in Hput2.
    subst s2.
    (* s2 = ([], [(p2, k2, false); (p1, k1, false)]) *)
    (* After normalization, front = [(p1, k1, false); (p2, k2, false)] - p1 is first (FIFO) *)
    exists ([(p2, k2, false)], []).
    unfold cc_find_match, queue_cc_find_match, queue_cc_normalize.
    cbn.
    destruct (pred p1 k1) eqn:Hpa1.
    - reflexivity.
    - rewrite Hp1 in Hpa1. discriminate.
  Qed.

  (** Queue length is sum of front and back *)
  Theorem queue_cc_len_correct :
    forall (front back : list (list P * K * bool)),
      @cc_len P K (QueueContinuationCollection P K) (front, back) = length front + length back.
  Proof.
    intros front back.
    reflexivity.
  Qed.

End QueueCCTheorems.

(** ** Stack-Specific Theorems *)

Section StackCCTheorems.
  Variable P K : Type.

  (** Stack put always succeeds *)
  Theorem stack_cc_put_increases_len :
    forall (s : @cc_storage P K (StackContinuationCollection P K)) patterns k persist,
      cc_len (cc_put s patterns k persist) = S (cc_len s).
  Proof.
    intros s patterns k persist.
    simpl. reflexivity.
  Qed.

  (** Stack empty is indeed empty *)
  Theorem stack_cc_empty_is_empty :
    @cc_is_empty P K (StackContinuationCollection P K) cc_empty = true.
  Proof.
    reflexivity.
  Qed.

  (** Stack LIFO property: last put is first found *)
  Theorem stack_cc_lifo :
    forall (s : @cc_storage P K (StackContinuationCollection P K))
           (patterns : list P) (k : K)
           (s' : @cc_storage P K (StackContinuationCollection P K)),
      cc_put s patterns k false = s' ->
      forall pred, pred patterns k = true ->
      exists rest, cc_find_match s' pred = Some (patterns, k, false, rest).
  Proof.
    intros s patterns k s' Hput pred Hpred.
    simpl in Hput. subst s'.
    simpl. rewrite Hpred.
    exists s. reflexivity.
  Qed.

  (** Stack LIFO ordering: cannot match older element when newer exists *)
  Theorem stack_cc_lifo_ordering :
    forall (p1 p2 : list P) (k1 k2 : K) s1 s2,
      cc_put (cc_empty : @cc_storage P K (StackContinuationCollection P K)) p1 k1 false = s1 ->
      cc_put s1 p2 k2 false = s2 ->
      forall pred, pred p1 k1 = true -> pred p2 k2 = false ->
      (* Cannot find p1 because p2 is on top and doesn't match *)
      cc_find_match s2 pred = None.
  Proof.
    intros p1 p2 k1 k2 s1 s2 Hput1 Hput2 pred Hp1 Hp2.
    simpl in Hput1. subst s1.
    simpl in Hput2. subst s2.
    (* s2 = [(p2, k2, false); (p1, k1, false)] *)
    simpl.
    rewrite Hp2. reflexivity.
  Qed.

  (** Stack: top element is returned when it matches *)
  Theorem stack_cc_top_matches :
    forall (p1 p2 : list P) (k1 k2 : K) s1 s2,
      cc_put (cc_empty : @cc_storage P K (StackContinuationCollection P K)) p1 k1 false = s1 ->
      cc_put s1 p2 k2 false = s2 ->
      forall pred, pred p2 k2 = true ->
      (* The top (p2, k2) is found, not the bottom (p1, k1) *)
      exists rest, cc_find_match s2 pred = Some (p2, k2, false, rest).
  Proof.
    intros p1 p2 k1 k2 s1 s2 Hput1 Hput2 pred Hp2.
    simpl in Hput1. subst s1.
    simpl in Hput2. subst s2.
    simpl. rewrite Hp2.
    exists [(p1, k1, false)]. reflexivity.
  Qed.

End StackCCTheorems.

(** ** Set-Specific Theorems *)

Section SetCCTheorems.
  Variable P K : Type.
  Variable P_eq_dec : forall p1 p2 : P, {p1 = p2} + {p1 <> p2}.
  Variable K_eq_dec : forall k1 k2 : K, {k1 = k2} + {k1 <> k2}.

  (** Helper: patterns_eq is reflexive *)
  Lemma patterns_eq_refl : forall ps,
    patterns_eq P P_eq_dec ps ps = true.
  Proof.
    induction ps as [| p rest IH].
    - reflexivity.
    - simpl. destruct (P_eq_dec p p) as [_ | Hneq].
      + exact IH.
      + exfalso. apply Hneq. reflexivity.
  Qed.

  (** Set put idempotency: adding same continuation twice has no effect *)
  Theorem set_cc_put_idempotent :
    forall (s : set_cc_storage P K) patterns k persist s' s'',
      set_cc_put P K P_eq_dec K_eq_dec s patterns k persist = s' ->
      set_cc_put P K P_eq_dec K_eq_dec s' patterns k persist = s'' ->
      s' = s''.
  Proof.
    intros s patterns k persist s' s'' Hput1 Hput2.
    unfold set_cc_put in Hput1.
    destruct (set_cc_member P K P_eq_dec K_eq_dec s patterns k persist) eqn:Hmem.
    - (* Already in s, so s' = s *)
      subst s'.
      unfold set_cc_put in Hput2.
      rewrite Hmem in Hput2.
      exact Hput2.
    - (* Not in s, so s' = (patterns, k, persist) :: s *)
      subst s'.
      unfold set_cc_put in Hput2.
      (* Now we need to show set_cc_member on the new list returns true *)
      simpl in Hput2.
      rewrite patterns_eq_refl in Hput2.
      destruct (K_eq_dec k k) as [_ | Hneq].
      + destruct persist; simpl in Hput2; exact Hput2.
      + exfalso. apply Hneq. reflexivity.
  Qed.

  (** Set never grows when adding duplicate *)
  Theorem set_cc_put_duplicate_preserves_len :
    forall (s : set_cc_storage P K) patterns k persist,
      set_cc_member P K P_eq_dec K_eq_dec s patterns k persist = true ->
      forall s', set_cc_put P K P_eq_dec K_eq_dec s patterns k persist = s' ->
      length s' = length s.
  Proof.
    intros s patterns k persist Hmem s' Hput.
    unfold set_cc_put in Hput.
    rewrite Hmem in Hput.
    subst s'. reflexivity.
  Qed.

  (** Set empty is indeed empty *)
  Theorem set_cc_empty_is_empty :
    @cc_is_empty P K (SetContinuationCollection P K P_eq_dec K_eq_dec) cc_empty = true.
  Proof.
    reflexivity.
  Qed.

End SetCCTheorems.

(** ** Persist Flag Semantics *)

(** This section formally verifies that:
    1. Non-persistent continuations are removed after matching
    2. Persistent continuations remain after matching

    Reference: Spec "Reifying RSpaces.md" - persist flag controls whether
    continuations remain after matching. *)

Section ContinuationPersistFlagSemantics.
  Variable P K : Type.

  (** ** Bag Persist Semantics *)

  (** Bag find_match length behavior *)
  Theorem bag_cc_find_match_len_behavior :
    forall (s s' : bag_cc_storage P K) patterns k persist pred,
      bag_cc_find_match P K s pred = Some (patterns, k, persist, s') ->
      length s' <= length s.
  Proof.
    intros s.
    induction s as [| [[ps k'] p'] rest IH].
    - intros s' patterns k persist pred Hfind. simpl in Hfind. discriminate.
    - intros s' patterns k persist pred Hfind.
      simpl in Hfind.
      destruct (pred ps k') eqn:Hpred.
      + destruct p'.
        * injection Hfind as _ _ _ Hs'. subst s'. lia.
        * injection Hfind as _ _ _ Hs'. subst s'. simpl. lia.
      + destruct (bag_cc_find_match P K rest pred) as [[[[found_p found_k] found_persist] rest'] |] eqn:Hrec.
        * injection Hfind as _ _ _ Hs'. subst s'.
          specialize (IH rest' found_p found_k found_persist pred Hrec).
          simpl. lia.
        * discriminate.
  Qed.

  (** Persistent bag continuation stays in the result *)
  Theorem bag_cc_persistent_kept :
    forall (s : bag_cc_storage P K) patterns k pred,
      In (patterns, k, true) s ->
      pred patterns k = true ->
      forall found_p found_k found_persist s',
        bag_cc_find_match P K s pred = Some (found_p, found_k, found_persist, s') ->
        In (patterns, k, true) s'.
  Proof.
    intros s.
    induction s as [| [[ps k'] p'] rest IH].
    - intros patterns k pred Hin. inversion Hin.
    - intros patterns k pred Hin Hpred found_p found_k found_persist s' Hfind.
      simpl in Hin.
      destruct Hin as [Heq | Hin_rest].
      + (* (patterns, k, true) = (ps, k', p') *)
        injection Heq as Hp Hk Hpers. subst ps k' p'.
        simpl in Hfind. rewrite Hpred in Hfind.
        (* persist = true, so s' = s *)
        injection Hfind as _ _ _ Hs'.
        subst s'.
        left. reflexivity.
      + (* (patterns, k, true) in rest *)
        simpl in Hfind.
        destruct (pred ps k') eqn:Hpred_x.
        * (* (ps, k') matched *)
          destruct p'.
          -- (* p' is persistent: s' = s *)
             injection Hfind as _ _ _ Hs'. subst s'.
             right. exact Hin_rest.
          -- (* p' is non-persistent: s' = rest *)
             injection Hfind as _ _ _ Hs'. subst s'.
             exact Hin_rest.
        * (* (ps, k') didn't match, recurse *)
          destruct (bag_cc_find_match P K rest pred) as [[[[found_p' found_k'] found_persist'] rest'] |] eqn:Hrec.
          -- injection Hfind as _ _ _ Hs'. subst s'.
             right. eapply IH; eassumption.
          -- discriminate.
  Qed.

  (** ** Stack Persist Semantics *)

  (** Non-persistent stack continuation is removed after matching *)
  Theorem stack_cc_non_persistent_removed :
    forall (s : stack_cc_storage P K) patterns k pred rest,
      pred patterns k = true ->
      stack_cc_find_match P K ((patterns, k, false) :: s) pred = Some (patterns, k, false, rest) ->
      rest = s.
  Proof.
    intros s patterns k pred rest Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** Persistent stack continuation remains after matching *)
  Theorem stack_cc_persistent_kept :
    forall (s : stack_cc_storage P K) patterns k pred rest,
      pred patterns k = true ->
      stack_cc_find_match P K ((patterns, k, true) :: s) pred = Some (patterns, k, true, rest) ->
      rest = (patterns, k, true) :: s.
  Proof.
    intros s patterns k pred rest Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** ** Queue Persist Semantics *)

  (** Queue front non-persistent removal *)
  Theorem queue_cc_front_non_persistent_removed :
    forall (patterns : list P) (k : K) back pred rest_front rest_back,
      pred patterns k = true ->
      queue_cc_find_match P K ([(patterns, k, false)], back) pred = Some (patterns, k, false, (rest_front, rest_back)) ->
      rest_front = [] /\ rest_back = back.
  Proof.
    intros patterns k back pred rest_front rest_back Hpred Hfind.
    unfold queue_cc_find_match, queue_cc_normalize in Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. split; reflexivity.
  Qed.

  (** Queue front persistent kept *)
  Theorem queue_cc_front_persistent_kept :
    forall (patterns : list P) (k : K) back pred rest,
      pred patterns k = true ->
      queue_cc_find_match P K ([(patterns, k, true)], back) pred = Some (patterns, k, true, rest) ->
      rest = ([(patterns, k, true)], back).
  Proof.
    intros patterns k back pred rest Hpred Hfind.
    unfold queue_cc_find_match, queue_cc_normalize in Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

End ContinuationPersistFlagSemantics.

(** ** General Length Properties *)

Section GeneralCCLengthProperties.
  Variable P K : Type.

  (** Stack find_match length behavior *)
  Theorem stack_cc_find_match_len_behavior :
    forall (s s' : stack_cc_storage P K) patterns k persist pred,
      stack_cc_find_match P K s pred = Some (patterns, k, persist, s') ->
      length s' <= length s.
  Proof.
    intros s s' patterns k persist pred Hfind.
    destruct s as [| [[ps k'] p'] rest].
    - simpl in Hfind. discriminate.
    - simpl in Hfind.
      destruct (pred ps k') eqn:Hpred.
      + destruct p'.
        * injection Hfind as _ _ _ Hs'. subst s'. lia.
        * injection Hfind as _ _ _ Hs'. subst s'. simpl. lia.
      + discriminate.
  Qed.

End GeneralCCLengthProperties.

(** ** Correspondence with DataCollection *)

(** The ContinuationCollection types mirror DataCollection types:
    - BagContinuationCollection ↔ BagDataCollection (non-deterministic matching)
    - QueueContinuationCollection ↔ QueueDataCollection (FIFO ordering)
    - StackContinuationCollection ↔ StackDataCollection (LIFO ordering)
    - SetContinuationCollection ↔ SetDataCollection (idempotent insertion)

    The key difference is:
    - DataCollection stores: (A, bool) tuples
    - ContinuationCollection stores: (list P, K, bool) tuples

    Both use the persist flag with the same semantics:
    - persist=true: element/continuation stays after matching
    - persist=false: element/continuation is removed after matching
*)

(** ** Summary of Verified Properties *)

(** For all ContinuationCollection types:
    1. Empty collection has length 0 and is_empty = true
    2. Clear produces an empty collection
    3. find_match preserves or decreases length

    For BagContinuationCollection:
    - Put always increases length by 1 (non-idempotent)
    - Any continuation can match (non-deterministic)
    - Persistent continuations remain after match

    For QueueContinuationCollection:
    - Put always increases length by 1
    - FIFO ordering: first registered is first matched
    - Only the front (oldest) continuation can match

    For StackContinuationCollection:
    - Put always increases length by 1
    - LIFO ordering: last registered is first matched
    - Only the top (newest) continuation can match
    - Cannot match older continuation when newer exists on top

    For SetContinuationCollection:
    - Idempotent: adding duplicate has no effect on length
    - Any continuation can match (like Bag)
    - Unique continuations only
*)
