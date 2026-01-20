(** * Data Collection Specifications

    This module specifies the behavior of data collections at channels.
    Each collection type has different semantics for storing and matching data.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/collections.rs
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Data Collection Interface *)

(** Abstract specification of a data collection. *)
Class DataCollection (A : Type) := {
  (** The underlying storage type *)
  dc_storage : Type;

  (** Empty collection *)
  dc_empty : dc_storage;

  (** Insert data (may fail for Cell type) *)
  dc_put : dc_storage -> A -> option dc_storage;

  (** Find and optionally remove matching data *)
  dc_find_match : dc_storage -> (A -> bool) -> option (A * dc_storage);

  (** Peek at data matching predicate WITHOUT removing.
      Returns (data, index) where index is the position in the collection.
      Reference: Spec lines 360-366 *)
  dc_peek : dc_storage -> (A -> bool) -> option (A * nat);

  (** Get all data without removing *)
  dc_get_all : dc_storage -> list (A * bool);  (* (data, persist) *)

  (** Check if collection is empty *)
  dc_is_empty : dc_storage -> bool;

  (** Get the number of elements *)
  dc_len : dc_storage -> nat;
}.

(** ** Bag Collection (Multiset) *)

(** Bag stores multiple copies and any can match. *)
Section BagCollection.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  Definition bag_storage := list (A * bool).  (* (data, persist) *)

  Definition bag_empty : bag_storage := [].

  Definition bag_put (s : bag_storage) (a : A) : option bag_storage :=
    Some ((a, false) :: s).  (* Default non-persistent *)

  Definition bag_put_persist (s : bag_storage) (a : A) (persist : bool)
    : option bag_storage :=
    Some ((a, persist) :: s).

  Fixpoint bag_find_match (s : bag_storage) (pred : A -> bool)
    : option (A * bag_storage) :=
    match s with
    | [] => None
    | (a, persist) :: rest =>
      if pred a then
        if persist then
          (* Persistent data: return match but keep in collection *)
          Some (a, s)
        else
          (* Non-persistent: remove from collection *)
          Some (a, rest)
      else
        match bag_find_match rest pred with
        | None => None
        | Some (found, rest') => Some (found, (a, persist) :: rest')
        end
    end.

  Definition bag_is_empty (s : bag_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition bag_len (s : bag_storage) : nat := length s.

  (** Peek at matching data WITHOUT removing.
      Returns (data, index) where index is the 0-based position.
      Reference: Spec lines 360-366 *)
  Fixpoint bag_peek_aux (s : bag_storage) (pred : A -> bool) (idx : nat)
    : option (A * nat) :=
    match s with
    | [] => None
    | (a, _) :: rest =>
      if pred a then Some (a, idx)
      else bag_peek_aux rest pred (S idx)
    end.

  Definition bag_peek (s : bag_storage) (pred : A -> bool) : option (A * nat) :=
    bag_peek_aux s pred 0.

  (** Bag collection instance *)
  Global Instance BagDataCollection : DataCollection A := {
    dc_storage := bag_storage;
    dc_empty := bag_empty;
    dc_put := bag_put;
    dc_find_match := bag_find_match;
    dc_peek := bag_peek;
    dc_get_all := fun s => s;
    dc_is_empty := bag_is_empty;
    dc_len := bag_len;
  }.
End BagCollection.

(** ** Queue Collection (FIFO) *)

Section QueueCollection.
  Variable A : Type.

  (* Queue uses two lists for amortized O(1) operations *)
  Definition queue_storage := (list (A * bool) * list (A * bool))%type.

  Definition queue_empty : queue_storage := ([], []).

  Definition queue_put (s : queue_storage) (a : A) : option queue_storage :=
    let '(front, back) := s in
    Some (front, (a, false) :: back).

  (* Normalize: move back to front when front is empty *)
  Definition queue_normalize (s : queue_storage) : queue_storage :=
    let '(front, back) := s in
    match front with
    | [] => (rev back, [])
    | _ => s
    end.

  Definition queue_find_match (s : queue_storage) (pred : A -> bool)
    : option (A * queue_storage) :=
    let s' := queue_normalize s in
    let '(front, back) := s' in
    match front with
    | [] => None
    | (a, persist) :: rest =>
      if pred a then
        if persist then
          Some (a, s')
        else
          Some (a, (rest, back))
      else
        None  (* Only front can match in a queue *)
    end.

  Definition queue_is_empty (s : queue_storage) : bool :=
    let '(front, back) := s in
    match front, back with
    | [], [] => true
    | _, _ => false
    end.

  Definition queue_len (s : queue_storage) : nat :=
    let '(front, back) := s in
    length front + length back.

  (** Peek at matching data WITHOUT removing.
      Queue only allows peeking at the front element (FIFO).
      Index is always 0 if matched. *)
  Definition queue_peek (s : queue_storage) (pred : A -> bool)
    : option (A * nat) :=
    let s' := queue_normalize s in
    let '(front, _) := s' in
    match front with
    | [] => None
    | (a, _) :: _ =>
      if pred a then Some (a, 0)
      else None  (* Only front can be peeked in FIFO queue *)
    end.

  Global Instance QueueDataCollection : DataCollection A := {
    dc_storage := queue_storage;
    dc_empty := queue_empty;
    dc_put := queue_put;
    dc_find_match := queue_find_match;
    dc_peek := queue_peek;
    dc_get_all := fun s => let '(f, b) := queue_normalize s in f;
    dc_is_empty := queue_is_empty;
    dc_len := queue_len;
  }.
End QueueCollection.

(** ** Cell Collection (Exactly Once) *)

Section CellCollection.
  Variable A : Type.

  Definition cell_storage := option (A * bool).

  Definition cell_empty : cell_storage := None.

  (* Cell fails if already full *)
  Definition cell_put (s : cell_storage) (a : A) : option cell_storage :=
    match s with
    | None => Some (Some (a, false))
    | Some _ => None  (* Error: cell already full *)
    end.

  Definition cell_find_match (s : cell_storage) (pred : A -> bool)
    : option (A * cell_storage) :=
    match s with
    | None => None
    | Some (a, persist) =>
      if pred a then
        if persist then
          Some (a, s)
        else
          Some (a, None)
      else
        None
    end.

  Definition cell_is_empty (s : cell_storage) : bool :=
    match s with
    | None => true
    | Some _ => false
    end.

  Definition cell_len (s : cell_storage) : nat :=
    match s with
    | None => 0
    | Some _ => 1
    end.

  (** Peek at the cell's data WITHOUT removing.
      Index is always 0 since cell holds at most one element. *)
  Definition cell_peek (s : cell_storage) (pred : A -> bool)
    : option (A * nat) :=
    match s with
    | None => None
    | Some (a, _) =>
      if pred a then Some (a, 0)
      else None
    end.

  Global Instance CellDataCollection : DataCollection A := {
    dc_storage := cell_storage;
    dc_empty := cell_empty;
    dc_put := cell_put;
    dc_find_match := cell_find_match;
    dc_peek := cell_peek;
    dc_get_all := fun s => match s with None => [] | Some x => [x] end;
    dc_is_empty := cell_is_empty;
    dc_len := cell_len;
  }.
End CellCollection.

(** ** Properties *)

Section DataCollectionProperties.
  Context {A : Type} `{DC : DataCollection A}.

  (** Empty collection has length 0 *)
  Definition empty_has_len_zero : Prop :=
    dc_len dc_empty = 0.

  (** Empty collection is empty *)
  Definition empty_is_empty : Prop :=
    dc_is_empty dc_empty = true.

  (** Put increases length by 1 (when successful) *)
  Definition put_increases_len : Prop :=
    forall s a s',
      dc_put s a = Some s' ->
      dc_len s' = S (dc_len s).

  (** Find match doesn't increase length *)
  Definition find_match_preserves_or_decreases_len : Prop :=
    forall s pred a s',
      dc_find_match s pred = Some (a, s') ->
      dc_len s' <= dc_len s.
End DataCollectionProperties.

(** ** Bag-Specific Theorems *)

Section BagTheorems.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  (** Bag put always succeeds *)
  Theorem bag_put_always_succeeds :
    forall (s : @dc_storage A (BagDataCollection A)) (a : A),
      exists s', dc_put s a = Some s'.
  Proof.
    intros s a.
    simpl. exists ((a, false) :: s). reflexivity.
  Qed.

  (** Bag empty is indeed empty *)
  Theorem bag_empty_is_empty :
    @dc_is_empty A (BagDataCollection A) dc_empty = true.
  Proof.
    reflexivity.
  Qed.
End BagTheorems.

(** ** Cell-Specific Theorems *)

Section CellTheorems.
  Variable A : Type.

  (** Cell enforces exactly-once semantics *)
  Theorem cell_put_fails_when_full :
    forall (a1 a2 : A) (persist : bool),
      @dc_put A (CellDataCollection A) (Some (a1, persist)) a2 = None.
  Proof.
    intros. reflexivity.
  Qed.

  (** Cell has at most one element *)
  Theorem cell_len_at_most_one :
    forall (s : @dc_storage A (CellDataCollection A)),
      dc_len s <= 1.
  Proof.
    intros s.
    destruct s as [| [a p]]; simpl; lia.
  Qed.
End CellTheorems.

(** ** Queue-Specific Theorems *)

Section QueueTheorems.
  Variable A : Type.

  (** Queue put always succeeds *)
  Theorem queue_put_always_succeeds :
    forall (s : @dc_storage A (QueueDataCollection A)) (a : A),
      exists s', dc_put s a = Some s'.
  Proof.
    intros s a.
    destruct s as [front back].
    simpl. eexists. reflexivity.
  Qed.

  (** Queue empty is indeed empty *)
  Theorem queue_empty_is_empty :
    @dc_is_empty A (QueueDataCollection A) dc_empty = true.
  Proof.
    reflexivity.
  Qed.

  (** Queue FIFO property helper: normalize preserves order *)
  Lemma queue_normalize_front_order :
    forall front back,
      fst (queue_normalize A (front, back)) =
        match front with
        | [] => rev back
        | _ => front
        end.
  Proof.
    intros front back.
    unfold queue_normalize.
    destruct front; reflexivity.
  Qed.

  (** Queue FIFO: first put is first found (when queue was empty) *)
  Theorem queue_fifo_first_in_first_out :
    forall (a : A),
      forall s',
        dc_put (dc_empty : @dc_storage A (QueueDataCollection A)) a = Some s' ->
        forall pred, pred a = true ->
        exists rest, dc_find_match s' pred = Some (a, rest).
  Proof.
    intros a s' Hput pred Hpred.
    cbn in Hput.
    injection Hput as Hs'.
    subst s'.
    (* s' = ([], [(a, false)]) *)
    (* Directly compute dc_find_match on the concrete queue *)
    exists ([], []).
    unfold dc_find_match, queue_find_match, queue_normalize.
    cbn.
    destruct (pred a) eqn:Hpa.
    - reflexivity.
    - rewrite Hpred in Hpa. discriminate.
  Qed.

  (** Queue FIFO: second put goes behind first *)
  Theorem queue_fifo_ordering :
    forall (a1 a2 : A) s1 s2,
      dc_put (dc_empty : @dc_storage A (QueueDataCollection A)) a1 = Some s1 ->
      dc_put s1 a2 = Some s2 ->
      forall pred, pred a1 = true -> pred a2 = true ->
      (* Finding on s2 should return a1 first *)
      exists rest, dc_find_match s2 pred = Some (a1, rest).
  Proof.
    intros a1 a2 s1 s2 Hput1 Hput2 pred Ha1 Ha2.
    cbn in Hput1.
    injection Hput1 as Hs1. subst s1.
    cbn in Hput2.
    injection Hput2 as Hs2. subst s2.
    (* s2 = ([], [(a2, false); (a1, false)]) *)
    (* After normalization, front = [(a1, false); (a2, false)] - a1 is first (FIFO) *)
    exists ([(a2, false)], []).
    unfold dc_find_match, queue_find_match, queue_normalize.
    cbn.
    destruct (pred a1) eqn:Hpa1.
    - reflexivity.
    - rewrite Ha1 in Hpa1. discriminate.
  Qed.

  (** Queue length is sum of front and back *)
  Theorem queue_len_correct :
    forall (front back : list (A * bool)),
      @dc_len A (QueueDataCollection A) (front, back) = length front + length back.
  Proof.
    intros front back.
    reflexivity.
  Qed.

  (** Queue put increases length by 1 *)
  Theorem queue_put_increases_len :
    forall (s : @dc_storage A (QueueDataCollection A)) a s',
      dc_put s a = Some s' ->
      dc_len s' = S (dc_len s).
  Proof.
    intros [front back] a s' Hput.
    simpl in Hput.
    injection Hput as Hs'. subst s'.
    simpl. lia.
  Qed.

End QueueTheorems.

(** ** Cell Additional Theorems *)

Section CellAdditionalTheorems.
  Variable A : Type.

  (** Cell is empty after creation *)
  Theorem cell_empty_is_empty :
    @dc_is_empty A (CellDataCollection A) dc_empty = true.
  Proof.
    reflexivity.
  Qed.

  (** Cell put on empty succeeds *)
  Theorem cell_put_on_empty_succeeds :
    forall (a : A),
      exists s', @dc_put A (CellDataCollection A) dc_empty a = Some s'.
  Proof.
    intros a.
    simpl. eexists. reflexivity.
  Qed.

  (** Cell after put is non-empty *)
  Theorem cell_after_put_nonempty :
    forall (a : A) s',
      @dc_put A (CellDataCollection A) dc_empty a = Some s' ->
      dc_is_empty s' = false.
  Proof.
    intros a s' Hput.
    simpl in Hput.
    injection Hput as Hs'. subst s'.
    reflexivity.
  Qed.

  (** Cell find_match succeeds on non-empty cell when predicate matches *)
  Theorem cell_find_match_succeeds :
    forall (a : A) persist pred,
      pred a = true ->
      exists rest, @dc_find_match A (CellDataCollection A) (Some (a, persist)) pred = Some (a, rest).
  Proof.
    intros a persist pred Hpred.
    simpl. rewrite Hpred.
    destruct persist.
    - eexists. reflexivity.
    - eexists. reflexivity.
  Qed.

  (** Cell find_match fails on empty cell *)
  Theorem cell_find_match_on_empty_fails :
    forall pred,
      @dc_find_match A (CellDataCollection A) dc_empty pred = None.
  Proof.
    intros pred. reflexivity.
  Qed.

  (** Cell exactly-once: put then put fails *)
  Theorem cell_exactly_once :
    forall (a1 a2 : A) s',
      @dc_put A (CellDataCollection A) dc_empty a1 = Some s' ->
      @dc_put A (CellDataCollection A) s' a2 = None.
  Proof.
    intros a1 a2 s' Hput.
    simpl in Hput.
    injection Hput as Hs'. subst s'.
    reflexivity.
  Qed.

End CellAdditionalTheorems.

(** ** Stack Collection (LIFO) *)

(** Stack stores data with last-in-first-out semantics.
    Only the top element can match. *)
Section StackCollection.
  Variable A : Type.

  Definition stack_storage := list (A * bool).  (* (data, persist) *)

  Definition stack_empty : stack_storage := [].

  Definition stack_put (s : stack_storage) (a : A) : option stack_storage :=
    Some ((a, false) :: s).  (* Push to top *)

  Definition stack_put_persist (s : stack_storage) (a : A) (persist : bool)
    : option stack_storage :=
    Some ((a, persist) :: s).

  (** Stack find only matches the top element *)
  Definition stack_find_match (s : stack_storage) (pred : A -> bool)
    : option (A * stack_storage) :=
    match s with
    | [] => None
    | (a, persist) :: rest =>
      if pred a then
        if persist then Some (a, s)
        else Some (a, rest)
      else None  (* Only top can match in a stack *)
    end.

  Definition stack_is_empty (s : stack_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition stack_len (s : stack_storage) : nat := length s.

  (** Peek at matching data WITHOUT removing.
      Stack only allows peeking at the top element (LIFO).
      Index is always 0 if matched. *)
  Definition stack_peek (s : stack_storage) (pred : A -> bool)
    : option (A * nat) :=
    match s with
    | [] => None
    | (a, _) :: _ =>
      if pred a then Some (a, 0)
      else None  (* Only top can be peeked in LIFO stack *)
    end.

  Global Instance StackDataCollection : DataCollection A := {
    dc_storage := stack_storage;
    dc_empty := stack_empty;
    dc_put := stack_put;
    dc_find_match := stack_find_match;
    dc_peek := stack_peek;
    dc_get_all := fun s => s;
    dc_is_empty := stack_is_empty;
    dc_len := stack_len;
  }.
End StackCollection.

(** ** Set Collection (Idempotent) *)

(** Set stores unique elements. Adding a duplicate is a no-op. *)
Section SetCollection.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  Definition set_storage := list (A * bool).

  Definition set_empty : set_storage := [].

  (** Check if element exists in set *)
  Fixpoint set_member (s : set_storage) (a : A) : bool :=
    match s with
    | [] => false
    | (x, _) :: rest =>
      if A_eq_dec x a then true else set_member rest a
    end.

  (** Set put is idempotent - adding existing element does nothing *)
  Definition set_put (s : set_storage) (a : A) : option set_storage :=
    if set_member s a then Some s  (* Already exists, idempotent *)
    else Some ((a, false) :: s).

  Definition set_put_persist (s : set_storage) (a : A) (persist : bool)
    : option set_storage :=
    if set_member s a then Some s
    else Some ((a, persist) :: s).

  (** Set find can match any element *)
  Fixpoint set_find_match (s : set_storage) (pred : A -> bool)
    : option (A * set_storage) :=
    match s with
    | [] => None
    | (a, persist) :: rest =>
      if pred a then
        if persist then Some (a, s)
        else Some (a, rest)
      else
        match set_find_match rest pred with
        | None => None
        | Some (found, rest') => Some (found, (a, persist) :: rest')
        end
    end.

  Definition set_is_empty (s : set_storage) : bool :=
    match s with
    | [] => true
    | _ => false
    end.

  Definition set_len (s : set_storage) : nat := length s.

  (** Peek at matching data WITHOUT removing.
      Set can peek at any matching element (like Bag).
      Returns (data, index) where index is the 0-based position. *)
  Fixpoint set_peek_aux (s : set_storage) (pred : A -> bool) (idx : nat)
    : option (A * nat) :=
    match s with
    | [] => None
    | (a, _) :: rest =>
      if pred a then Some (a, idx)
      else set_peek_aux rest pred (S idx)
    end.

  Definition set_peek (s : set_storage) (pred : A -> bool) : option (A * nat) :=
    set_peek_aux s pred 0.

  Global Instance SetDataCollection : DataCollection A := {
    dc_storage := set_storage;
    dc_empty := set_empty;
    dc_put := set_put;
    dc_find_match := set_find_match;
    dc_peek := set_peek;
    dc_get_all := fun s => s;
    dc_is_empty := set_is_empty;
    dc_len := set_len;
  }.
End SetCollection.

(** ** Stack-Specific Theorems *)

Section StackTheorems.
  Variable A : Type.

  (** Stack put always succeeds *)
  Theorem stack_put_always_succeeds :
    forall (s : @dc_storage A (StackDataCollection A)) (a : A),
      exists s', dc_put s a = Some s'.
  Proof.
    intros s a.
    simpl. exists ((a, false) :: s). reflexivity.
  Qed.

  (** Stack LIFO property: last put is first found *)
  Theorem stack_lifo :
    forall (s : @dc_storage A (StackDataCollection A)) (a : A) (s' : @dc_storage A (StackDataCollection A)),
      dc_put s a = Some s' ->
      forall pred, pred a = true ->
      exists rest, dc_find_match s' pred = Some (a, rest).
  Proof.
    intros s a s' Hput pred Hpred.
    simpl in Hput. injection Hput as Hs'.
    rewrite <- Hs'. simpl.
    rewrite Hpred.
    exists s. reflexivity.
  Qed.
End StackTheorems.

(** ** Set-Specific Theorems *)

Section SetTheorems.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  (** Set put idempotency: adding same element twice has no effect *)
  Theorem set_put_idempotent :
    forall (s : set_storage A) (a : A) s' s'',
      set_put A A_eq_dec s a = Some s' ->
      set_put A A_eq_dec s' a = Some s'' ->
      s' = s''.
  Proof.
    intros s a s' s'' Hput1 Hput2.
    unfold set_put in Hput1.
    destruct (set_member A A_eq_dec s a) eqn:Hmem.
    - (* a already in s, so s' = s *)
      injection Hput1 as Hs'. subst s'.
      (* Now Hput2 : set_put A A_eq_dec s a = Some s'' *)
      unfold set_put in Hput2.
      rewrite Hmem in Hput2.
      injection Hput2 as Hs''. exact Hs''.
    - (* a not in s, so s' = (a, false) :: s *)
      injection Hput1 as Hs'. subst s'.
      (* Now s' = (a, false) :: s, and we need to show set_put on this returns same *)
      unfold set_put in Hput2.
      (* set_member on ((a, false) :: s) for a should be true *)
      simpl in Hput2.
      destruct (A_eq_dec a a) as [_ | Hneq].
      + (* a = a, so set_member returns true, so s'' = (a, false) :: s *)
        injection Hput2 as Hs''. exact Hs''.
      + exfalso. apply Hneq. reflexivity.
  Qed.

  (** Set never grows when adding duplicate *)
  Theorem set_put_duplicate_preserves_len :
    forall (s : set_storage A) (a : A),
      set_member A A_eq_dec s a = true ->
      forall s', set_put A A_eq_dec s a = Some s' -> set_len A s' = set_len A s.
  Proof.
    intros s a Hmem s' Hput.
    unfold set_put in Hput.
    rewrite Hmem in Hput.
    injection Hput as Hs'.
    rewrite <- Hs'. reflexivity.
  Qed.
End SetTheorems.

(** ** Persist Flag Semantics *)

(** This section formally verifies that:
    1. Non-persistent data is removed after matching
    2. Persistent data remains after matching

    Reference: Spec "Reifying RSpaces.md" - persist flag controls whether
    data remains after matching. *)

Section PersistFlagSemantics.
  Variable A : Type.

  (** ** Bag Persist Semantics *)

  (** Non-persistent bag data is removed after matching *)
  Theorem bag_non_persistent_removed :
    forall (s : bag_storage A) (a : A) pred s',
      In (a, false) s ->
      pred a = true ->
      bag_find_match A s pred = Some (a, s') ->
      ~ In (a, false) s'.
  Proof.
    intros s a pred s' Hin Hpred Hfind.
    induction s as [| [x p] rest IH].
    - (* s = [] *)
      inversion Hin.
    - (* s = (x, p) :: rest *)
      simpl in Hin. destruct Hin as [Heq | Hin_rest].
      + (* (a, false) = (x, p) *)
        injection Heq as Ha Hp. subst x p.
        simpl in Hfind.
        rewrite Hpred in Hfind.
        (* persist = false, so result is rest *)
        injection Hfind as Hs'. subst s'.
        (* Need to show ~ In (a, false) rest - this is NOT true in general!
           The element could appear multiple times in a bag. *)
        (* This theorem as stated is too strong for bags with duplicates.
           Let's refine: the MATCHED instance is removed. *)
  Abort.

  (** Refined theorem: bag_find_match removes the first matching instance *)
  Theorem bag_find_match_removes_first :
    forall (s : bag_storage A) (a : A) pred s',
      bag_find_match A s pred = Some (a, s') ->
      (* The match was non-persistent (persist = false in the result) *)
      exists prefix suffix,
        s = prefix ++ [(a, false)] ++ suffix /\
        s' = prefix ++ suffix /\
        (* No element in prefix matches pred *)
        (forall x p, In (x, p) prefix -> pred x = false).
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - (* s = [] *)
      intros a pred s' Hfind. simpl in Hfind. discriminate.
    - (* s = (x, p) :: rest *)
      intros a pred s' Hfind.
      simpl in Hfind.
      destruct (pred x) eqn:Hpred_x.
      + (* pred x = true: this is the match *)
        destruct p.
        * (* persist = true: keeps the data *)
          injection Hfind as Ha Hs'.
          subst a s'.
          (* When persist = true, the theorem doesn't apply: data is kept *)
          (* The result is Some (x, (x, true) :: rest), not removing anything *)
          (* So this case should result in s' = s, not removing the element *)
          (* Actually, when persist is true, the data is NOT removed! *)
          (* We need to exclude this case from our theorem *)
        Abort.

  (** Non-persistent bag data: the matched element is not in result when persist=false *)
  Theorem bag_find_match_non_persist_removes :
    forall (s : bag_storage A) (a : A) pred s',
      bag_find_match A s pred = Some (a, s') ->
      (* If the match was for a non-persistent element, then: *)
      length s' < length s \/
      (* Or the element was persistent and s' = s *)
      s' = s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros a pred s' Hfind. simpl in Hfind. discriminate.
    - intros a pred s' Hfind.
      simpl in Hfind.
      destruct (pred x) eqn:Hpred_x.
      + (* pred x = true: this is the match *)
        destruct p.
        * (* persist = true *)
          injection Hfind as Ha Hs'.
          subst a s'.
          right. reflexivity.
        * (* persist = false *)
          injection Hfind as Ha Hs'.
          subst a s'.
          left. simpl. lia.
      + (* pred x = false: recurse *)
        destruct (bag_find_match A rest pred) as [[found rest'] |] eqn:Hrec.
        * injection Hfind as Ha Hs'.
          subst a s'.
          specialize (IH found pred rest' Hrec).
          destruct IH as [Hlt | Heq].
          -- left. simpl. lia.
          -- right. rewrite Heq. reflexivity.
        * discriminate.
  Qed.

  (** Simpler formulation: Persistent data stays in the result *)
  Theorem bag_persistent_kept :
    forall (s : bag_storage A) (a : A) pred,
      In (a, true) s ->
      pred a = true ->
      forall found s',
        bag_find_match A s pred = Some (found, s') ->
        In (a, true) s'.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros a pred Hin. inversion Hin.
    - intros a pred Hin Hpred found s' Hfind.
      simpl in Hin.
      destruct Hin as [Heq | Hin_rest].
      + (* (a, true) = (x, p) *)
        injection Heq as Ha Hp. subst x p.
        simpl in Hfind. rewrite Hpred in Hfind.
        (* persist = true, so s' = s *)
        injection Hfind as Hfound Hs'.
        subst found s'.
        left. reflexivity.
      + (* (a, true) in rest *)
        simpl in Hfind.
        destruct (pred x) eqn:Hpred_x.
        * (* x matched *)
          destruct p.
          -- (* x is persistent: s' = s *)
             injection Hfind as _ Hs'. subst s'.
             right. exact Hin_rest.
          -- (* x is non-persistent: s' = rest *)
             injection Hfind as _ Hs'. subst s'.
             exact Hin_rest.
        * (* x didn't match, recurse *)
          destruct (bag_find_match A rest pred) as [[found' rest'] |] eqn:Hrec.
          -- injection Hfind as Hfound Hs'. subst found s'.
             right. eapply IH; eassumption.
          -- discriminate.
  Qed.

  (** ** Cell Persist Semantics *)

  (** Non-persistent cell data is removed after matching *)
  Theorem cell_non_persistent_removed :
    forall (a : A) pred s',
      pred a = true ->
      cell_find_match A (Some (a, false)) pred = Some (a, s') ->
      s' = None.
  Proof.
    intros a pred s' Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** Persistent cell data remains after matching *)
  Theorem cell_persistent_kept :
    forall (a : A) pred s',
      pred a = true ->
      cell_find_match A (Some (a, true)) pred = Some (a, s') ->
      s' = Some (a, true).
  Proof.
    intros a pred s' Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** ** Stack Persist Semantics *)

  (** Non-persistent stack data is removed after matching *)
  Theorem stack_non_persistent_removed :
    forall (s : stack_storage A) (a : A) pred rest,
      pred a = true ->
      stack_find_match A ((a, false) :: s) pred = Some (a, rest) ->
      rest = s.
  Proof.
    intros s a pred rest Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** Persistent stack data remains after matching *)
  Theorem stack_persistent_kept :
    forall (s : stack_storage A) (a : A) pred rest,
      pred a = true ->
      stack_find_match A ((a, true) :: s) pred = Some (a, rest) ->
      rest = (a, true) :: s.
  Proof.
    intros s a pred rest Hpred Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

  (** ** Queue Persist Semantics *)

  (** Queue persist semantics for the front element *)
  Theorem queue_front_non_persistent_removed :
    forall (a : A) back pred rest_front rest_back,
      pred a = true ->
      queue_find_match A ([(a, false)], back) pred = Some (a, (rest_front, rest_back)) ->
      rest_front = [] /\ rest_back = back.
  Proof.
    intros a back pred rest_front rest_back Hpred Hfind.
    unfold queue_find_match, queue_normalize in Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. split; reflexivity.
  Qed.

  Theorem queue_front_persistent_kept :
    forall (a : A) back pred rest,
      pred a = true ->
      queue_find_match A ([(a, true)], back) pred = Some (a, rest) ->
      rest = ([(a, true)], back).
  Proof.
    intros a back pred rest Hpred Hfind.
    unfold queue_find_match, queue_normalize in Hfind.
    simpl in Hfind.
    rewrite Hpred in Hfind.
    inversion Hfind. reflexivity.
  Qed.

End PersistFlagSemantics.

(** ** General Persist Flag Properties *)

Section GeneralPersistProperties.
  Variable A : Type.

  (** Property: length decreases when non-persistent match occurs *)
  Definition match_decreases_len_when_non_persistent (s s' : bag_storage A) (a : A) : Prop :=
    In (a, false) s ->
    length s' < length s.

  (** Property: length unchanged when persistent match occurs *)
  Definition match_preserves_len_when_persistent (s s' : bag_storage A) (a : A) : Prop :=
    s' = s ->
    length s' = length s.

  (** Bag find_match length behavior *)
  Theorem bag_find_match_len_behavior :
    forall (s s' : bag_storage A) (a : A) pred,
      bag_find_match A s pred = Some (a, s') ->
      length s' <= length s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros s' a pred Hfind. simpl in Hfind. discriminate.
    - intros s' a pred Hfind.
      simpl in Hfind.
      destruct (pred x) eqn:Hpred_x.
      + destruct p.
        * injection Hfind as _ Hs'. subst s'. lia.
        * injection Hfind as _ Hs'. subst s'. simpl. lia.
      + destruct (bag_find_match A rest pred) as [[found rest'] |] eqn:Hrec.
        * injection Hfind as _ Hs'. subst s'.
          specialize (IH rest' found pred Hrec).
          simpl. lia.
        * discriminate.
  Qed.

  (** Cell find_match length behavior *)
  Theorem cell_find_match_len_behavior :
    forall (s s' : cell_storage A) (a : A) pred,
      cell_find_match A s pred = Some (a, s') ->
      cell_len A s' <= cell_len A s.
  Proof.
    intros s s' a pred Hfind.
    destruct s as [[x p] |].
    - simpl in Hfind.
      destruct (pred x) eqn:Hpred.
      + destruct p.
        * injection Hfind as _ Hs'. subst s'. simpl. lia.
        * injection Hfind as _ Hs'. subst s'. simpl. lia.
      + discriminate.
    - simpl in Hfind. discriminate.
  Qed.

  (** Stack find_match length behavior *)
  Theorem stack_find_match_len_behavior :
    forall (s s' : stack_storage A) (a : A) pred,
      stack_find_match A s pred = Some (a, s') ->
      length s' <= length s.
  Proof.
    intros s s' a pred Hfind.
    destruct s as [| [x p] rest].
    - simpl in Hfind. discriminate.
    - simpl in Hfind.
      destruct (pred x) eqn:Hpred.
      + destruct p.
        * injection Hfind as _ Hs'. subst s'. lia.
        * injection Hfind as _ Hs'. subst s'. simpl. lia.
      + discriminate.
  Qed.

End GeneralPersistProperties.

(** ** Peek Semantics Theorems *)

(** This section formally verifies that:
    1. Peek does not modify the collection
    2. Peek is consistent with find_match (finds same data)

    Reference: Spec "Reifying RSpaces.md" lines 360-366 *)

Section PeekSemantics.
  Variable A : Type.

  (** ** Bag Peek Theorems *)

  (** Helper lemma for bag_peek_aux *)
  Lemma bag_peek_aux_bound :
    forall (s : bag_storage A) pred start a idx,
      bag_peek_aux A s pred start = Some (a, idx) ->
      start <= idx /\ idx < start + length s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst.
        simpl. lia.
      + specialize (IH pred (S start) a idx Hpeek).
        simpl. lia.
  Qed.

  (** Bag peek does not modify collection - index is valid *)
  Theorem bag_peek_index_valid :
    forall (s : bag_storage A) pred a idx,
      bag_peek A s pred = Some (a, idx) ->
      idx < length s.
  Proof.
    intros s pred a idx Hpeek.
    unfold bag_peek in Hpeek.
    apply bag_peek_aux_bound in Hpeek.
    lia.
  Qed.

  (** Helper lemma for bag_peek_finds_existing *)
  Lemma bag_peek_aux_finds_existing :
    forall (s : bag_storage A) pred start a idx,
      bag_peek_aux A s pred start = Some (a, idx) ->
      exists p, In (a, p) s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        exists p. left. reflexivity.
      + specialize (IH pred (S start) a idx Hpeek).
        destruct IH as [p' Hin].
        exists p'. right. exact Hin.
  Qed.

  (** Bag peek returns data that exists in collection *)
  Theorem bag_peek_finds_existing :
    forall (s : bag_storage A) pred a idx,
      bag_peek A s pred = Some (a, idx) ->
      exists p, In (a, p) s.
  Proof.
    intros s pred a idx Hpeek.
    unfold bag_peek in Hpeek.
    eapply bag_peek_aux_finds_existing. exact Hpeek.
  Qed.

  (** Helper lemma for bag_peek_consistent_with_find *)
  Lemma bag_peek_aux_consistent_with_find :
    forall (s : bag_storage A) pred start a idx,
      bag_peek_aux A s pred start = Some (a, idx) ->
      exists s', bag_find_match A s pred = Some (a, s').
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        simpl. rewrite Hpred.
        destruct p.
        * exists ((x, true) :: rest). reflexivity.
        * exists rest. reflexivity.
      + specialize (IH pred (S start) a idx Hpeek).
        destruct IH as [s' Hfind].
        simpl. rewrite Hpred.
        rewrite Hfind.
        eexists. reflexivity.
  Qed.

  (** Bag peek is consistent with find_match - both find the same data *)
  Theorem bag_peek_consistent_with_find :
    forall (s : bag_storage A) pred a idx,
      bag_peek A s pred = Some (a, idx) ->
      exists s', bag_find_match A s pred = Some (a, s').
  Proof.
    intros s pred a idx Hpeek.
    unfold bag_peek in Hpeek.
    eapply bag_peek_aux_consistent_with_find. exact Hpeek.
  Qed.

  (** ** Cell Peek Theorems *)

  (** Cell peek returns data that exists *)
  Theorem cell_peek_finds_existing :
    forall (s : cell_storage A) pred a idx,
      cell_peek A s pred = Some (a, idx) ->
      exists p, s = Some (a, p).
  Proof.
    intros s pred a idx Hpeek.
    destruct s as [[x p] |].
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        exists p. reflexivity.
      + discriminate.
    - simpl in Hpeek. discriminate.
  Qed.

  (** Cell peek consistent with find *)
  Theorem cell_peek_consistent_with_find :
    forall (s : cell_storage A) pred a idx,
      cell_peek A s pred = Some (a, idx) ->
      exists s', cell_find_match A s pred = Some (a, s').
  Proof.
    intros s pred a idx Hpeek.
    destruct s as [[x p] |].
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        simpl. rewrite Hpred.
        destruct p.
        * exists (Some (x, true)). reflexivity.
        * exists None. reflexivity.
      + discriminate.
    - simpl in Hpeek. discriminate.
  Qed.

  (** ** Stack Peek Theorems *)

  (** Stack peek returns the top element *)
  Theorem stack_peek_is_top :
    forall (s : stack_storage A) pred a idx,
      stack_peek A s pred = Some (a, idx) ->
      idx = 0 /\ exists p rest, s = (a, p) :: rest.
  Proof.
    intros s pred a idx Hpeek.
    destruct s as [| [x p] rest].
    - simpl in Hpeek. discriminate.
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a idx.
        split. reflexivity.
        exists p, rest. reflexivity.
      + discriminate.
  Qed.

  (** Stack peek consistent with find *)
  Theorem stack_peek_consistent_with_find :
    forall (s : stack_storage A) pred a idx,
      stack_peek A s pred = Some (a, idx) ->
      exists s', stack_find_match A s pred = Some (a, s').
  Proof.
    intros s pred a idx Hpeek.
    destruct s as [| [x p] rest].
    - simpl in Hpeek. discriminate.
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        simpl. rewrite Hpred.
        destruct p.
        * exists ((x, true) :: rest). reflexivity.
        * exists rest. reflexivity.
      + discriminate.
  Qed.

  (** ** Queue Peek Theorems *)

  (** Queue peek returns the front element *)
  Theorem queue_peek_is_front :
    forall (s : queue_storage A) pred a idx,
      queue_peek A s pred = Some (a, idx) ->
      idx = 0.
  Proof.
    intros s pred a idx Hpeek.
    unfold queue_peek in Hpeek.
    destruct (queue_normalize A s) as [front back] eqn:Hnorm.
    destruct front as [| [x p] rest].
    - simpl in Hpeek. discriminate.
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as _ Hidx. symmetry. exact Hidx.
      + discriminate.
  Qed.

  (** Queue peek consistent with find *)
  Theorem queue_peek_consistent_with_find :
    forall (s : queue_storage A) pred a idx,
      queue_peek A s pred = Some (a, idx) ->
      exists s', queue_find_match A s pred = Some (a, s').
  Proof.
    intros s pred a idx Hpeek.
    unfold queue_peek in Hpeek.
    destruct (queue_normalize A s) as [front back] eqn:Hnorm.
    destruct front as [| [x p] rest].
    - simpl in Hpeek. discriminate.
    - simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        unfold queue_find_match.
        rewrite Hnorm. simpl. rewrite Hpred.
        destruct p.
        * exists ((x, true) :: rest, back). reflexivity.
        * exists (rest, back). reflexivity.
      + discriminate.
  Qed.

  (** ** Set Peek Theorems *)

  (** Helper lemma for set_peek_aux.
      Note: set_peek_aux does not use A_eq_dec, so it's not a parameter. *)
  Lemma set_peek_aux_bound :
    forall (s : set_storage A) pred start a idx,
      set_peek_aux A s pred start = Some (a, idx) ->
      start <= idx /\ idx < start + length s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst.
        simpl. lia.
      + specialize (IH pred (S start) a idx Hpeek).
        simpl. lia.
  Qed.

  (** Set peek index is valid.
      Note: set_peek doesn't use A_eq_dec internally. *)
  Theorem set_peek_index_valid :
    forall (eqdec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2})
           (s : set_storage A) pred a idx,
      @dc_peek A (SetDataCollection A eqdec) s pred = Some (a, idx) ->
      idx < length s.
  Proof.
    intros eqdec s pred a idx Hpeek.
    simpl in Hpeek. unfold set_peek in Hpeek.
    apply set_peek_aux_bound in Hpeek.
    lia.
  Qed.

  (** Helper lemma for set_peek_finds_existing *)
  Lemma set_peek_aux_finds_existing :
    forall (s : set_storage A) pred start a idx,
      set_peek_aux A s pred start = Some (a, idx) ->
      exists p, In (a, p) s.
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        exists p. left. reflexivity.
      + specialize (IH pred (S start) a idx Hpeek).
        destruct IH as [p' Hin].
        exists p'. right. exact Hin.
  Qed.

  (** Set peek finds existing data *)
  Theorem set_peek_finds_existing :
    forall (eqdec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2})
           (s : set_storage A) pred a idx,
      @dc_peek A (SetDataCollection A eqdec) s pred = Some (a, idx) ->
      exists p, In (a, p) s.
  Proof.
    intros eqdec s pred a idx Hpeek.
    simpl in Hpeek. unfold set_peek in Hpeek.
    eapply set_peek_aux_finds_existing. exact Hpeek.
  Qed.

  (** Helper lemma for set_peek_consistent_with_find.
      Note: set_find_match also doesn't use A_eq_dec. *)
  Lemma set_peek_aux_consistent_with_find :
    forall (s : set_storage A) pred start a idx,
      set_peek_aux A s pred start = Some (a, idx) ->
      exists s', set_find_match A s pred = Some (a, s').
  Proof.
    intros s.
    induction s as [| [x p] rest IH].
    - intros pred start a idx Hpeek. simpl in Hpeek. discriminate.
    - intros pred start a idx Hpeek.
      simpl in Hpeek.
      destruct (pred x) eqn:Hpred.
      + injection Hpeek as Ha Hidx. subst a.
        simpl. rewrite Hpred.
        destruct p.
        * exists ((x, true) :: rest). reflexivity.
        * exists rest. reflexivity.
      + specialize (IH pred (S start) a idx Hpeek).
        destruct IH as [s' Hfind].
        simpl. rewrite Hpred.
        rewrite Hfind.
        eexists. reflexivity.
  Qed.

  (** Set peek consistent with find *)
  Theorem set_peek_consistent_with_find :
    forall (eqdec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2})
           (s : set_storage A) pred a idx,
      @dc_peek A (SetDataCollection A eqdec) s pred = Some (a, idx) ->
      exists s', @dc_find_match A (SetDataCollection A eqdec) s pred = Some (a, s').
  Proof.
    intros eqdec s pred a idx Hpeek.
    simpl in Hpeek. unfold set_peek in Hpeek.
    simpl. eapply set_peek_aux_consistent_with_find. exact Hpeek.
  Qed.

End PeekSemantics.
