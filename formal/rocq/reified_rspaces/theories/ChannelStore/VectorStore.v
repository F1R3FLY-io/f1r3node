(** * Vector Channel Store Specification

    This module specifies the behavior of Vector-based channel stores.
    Vector stores use a dynamically growing array indexed by natural numbers.

    Reference: Rust implementation in
      rspace++/src/rspace/storage/vector_store.rs
*)

From Stdlib Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
From ReifiedRSpaces.Collections Require Import DataCollection.
Import ListNotations.

(** ** Vector Store Types *)

Section VectorStore.
  Variable A K : Type.

  (** Vector storage uses a list where index = channel number *)
  Definition vector_data_store := list (list (A * bool)).
  Definition vector_cont_store := list (list (list A * K * bool)).

  Record vector_storage := mkVectorStorage {
    vs_data : vector_data_store;
    vs_conts : vector_cont_store;
  }.

  (** Empty vector store *)
  Definition vector_empty : vector_storage := {|
    vs_data := [];
    vs_conts := [];
  |}.

  (** Extend vector to accommodate index *)
  Fixpoint extend_to {X} (default : X) (l : list X) (n : nat) : list X :=
    match n with
    | 0 => match l with
           | [] => [default]
           | _ => l
           end
    | S n' => match l with
              | [] => extend_to default [default] n'
              | x :: rest => x :: extend_to default rest n'
              end
    end.

  (** Get data at channel index *)
  Definition vector_get_data (s : vector_storage) (c : nat) : list (A * bool) :=
    nth c (vs_data s) [].

  (** Put data at channel index, extending if necessary *)
  Definition vector_put_data (s : vector_storage) (c : nat) (a : A) (persist : bool)
    : vector_storage :=
    let data := vs_data s in
    let extended := extend_to [] data c in
    let current := nth c extended [] in
    let updated := (a, persist) :: current in
    {|
      vs_data := firstn c extended ++ [updated] ++ skipn (S c) extended;
      vs_conts := vs_conts s;
    |}.

  (** Remove first matching data from channel *)
  Fixpoint remove_first_match_data (l : list (A * bool)) (pred : A -> bool)
    : option (A * list (A * bool)) :=
    match l with
    | [] => None
    | (a, p) :: rest =>
      if pred a then Some (a, rest)
      else match remove_first_match_data rest pred with
           | None => None
           | Some (found, rest') => Some (found, (a, p) :: rest')
           end
    end.

  Definition vector_remove_data (s : vector_storage) (c : nat) (pred : A -> bool)
    : option (A * vector_storage) :=
    let data := vector_get_data s c in
    match remove_first_match_data data pred with
    | None => None
    | Some (found, rest) =>
      let vs := vs_data s in
      let extended := extend_to [] vs c in
      Some (found, {|
        vs_data := firstn c extended ++ [rest] ++ skipn (S c) extended;
        vs_conts := vs_conts s;
      |})
    end.

  (** Gensym returns next index and extends the vector *)
  Definition vector_gensym (s : vector_storage) : nat * vector_storage :=
    let idx := length (vs_data s) in
    (idx, {|
      vs_data := vs_data s ++ [[]];  (* Append empty slot *)
      vs_conts := vs_conts s;
    |}).

  (** Clear all data *)
  Definition vector_clear (s : vector_storage) : vector_storage := {|
    vs_data := [];
    vs_conts := [];
  |}.

  (** Check if empty *)
  Definition vector_is_empty (s : vector_storage) : bool :=
    match vs_data s with
    | [] => true
    | _ => false
    end.

  (** Snapshot for checkpointing *)
  Definition vector_snapshot (s : vector_storage) : vector_storage := s.

End VectorStore.

(** ** Vector Store Properties *)

Section VectorStoreProperties.
  Variable A K : Type.

  (** Empty store has no data *)
  Theorem vector_empty_no_data :
    forall (c : nat),
      vector_get_data A K (vector_empty A K) c = [].
  Proof.
    intros c.
    unfold vector_get_data, vector_empty.
    simpl.
    destruct c; reflexivity.
  Qed.

  (** Gensym returns the current length *)
  Theorem vector_gensym_returns_length :
    forall (s : vector_storage A K) (c : nat) (s' : vector_storage A K),
      vector_gensym A K s = (c, s') ->
      c = length (vs_data A K s).
  Proof.
    intros s c s' H.
    unfold vector_gensym in H.
    injection H as Hc Hs'.
    symmetry. exact Hc.
  Qed.

  (** Gensym strictly increases the length *)
  Theorem vector_gensym_increases_length :
    forall (s : vector_storage A K) (c : nat) (s' : vector_storage A K),
      vector_gensym A K s = (c, s') ->
      length (vs_data A K s') = S (length (vs_data A K s)).
  Proof.
    intros s c s' H.
    unfold vector_gensym in H.
    injection H as Hc Hs'.
    subst s'.
    simpl.
    rewrite length_app.
    simpl. lia.
  Qed.

  (** Sequential gensym returns distinct indices (monotonicity) *)
  Theorem vector_gensym_monotonic :
    forall (s : vector_storage A K)
           (c1 : nat) (s1 : vector_storage A K)
           (c2 : nat) (s2 : vector_storage A K),
      vector_gensym A K s = (c1, s1) ->
      vector_gensym A K s1 = (c2, s2) ->
      c2 = S c1.
  Proof.
    intros s c1 s1 c2 s2 H1 H2.
    apply vector_gensym_returns_length in H1 as Hc1.
    apply vector_gensym_returns_length in H2 as Hc2.
    apply vector_gensym_increases_length in H1 as Hlen.
    subst c1 c2.
    rewrite Hlen. reflexivity.
  Qed.

  (** Sequential gensym returns distinct indices *)
  Theorem vector_gensym_distinct :
    forall (s : vector_storage A K)
           (c1 : nat) (s1 : vector_storage A K)
           (c2 : nat) (s2 : vector_storage A K),
      vector_gensym A K s = (c1, s1) ->
      vector_gensym A K s1 = (c2, s2) ->
      c1 <> c2.
  Proof.
    intros s c1 s1 c2 s2 H1 H2.
    pose proof (vector_gensym_monotonic s c1 s1 c2 s2 H1 H2) as Hmono.
    rewrite Hmono. lia.
  Qed.

  (** Gensym preserves existing data *)
  Theorem vector_gensym_preserves_data :
    forall (s : vector_storage A K) (c' : nat) (s' : vector_storage A K) (c : nat),
      vector_gensym A K s = (c', s') ->
      c < length (vs_data A K s) ->
      vector_get_data A K s' c = vector_get_data A K s c.
  Proof.
    intros s c' s' c H Hlt.
    unfold vector_gensym in H.
    injection H as Hc' Hs'.
    subst s'.
    unfold vector_get_data.
    simpl.
    (* nth c (vs_data s ++ [[]]) [] = nth c (vs_data s) [] *)
    rewrite app_nth1.
    - reflexivity.
    - exact Hlt.
  Qed.

  (** Clear removes all data *)
  Theorem vector_clear_empty :
    forall (s : vector_storage A K),
      vector_is_empty A K (vector_clear A K s) = true.
  Proof.
    intros s.
    unfold vector_is_empty, vector_clear.
    simpl. reflexivity.
  Qed.

  (** Helper: remove_first_match_data returns element satisfying predicate *)
  Lemma remove_first_match_data_pred :
    forall (l : list (A * bool)) (pred : A -> bool) (found : A) (rest : list (A * bool)),
      remove_first_match_data A l pred = Some (found, rest) ->
      pred found = true.
  Proof.
    induction l as [| [h hp] t IH].
    - intros. discriminate.
    - intros pred0 found rest Hfind.
      simpl in Hfind.
      destruct (pred0 h) eqn:Hpred.
      + inversion Hfind; subst. exact Hpred.
      + destruct (remove_first_match_data A t pred0) as [[f' r'] |] eqn:Hrec.
        * inversion Hfind; subst.
          eapply IH. exact Hrec.
        * discriminate.
  Qed.

  (** Remove returns matching element *)
  Theorem vector_remove_data_returns_match :
    forall (s : vector_storage A K) (c : nat) (pred : A -> bool)
           (a : A) (s' : vector_storage A K),
      vector_remove_data A K s c pred = Some (a, s') ->
      pred a = true.
  Proof.
    intros s c pred a s' Hremove.
    unfold vector_remove_data in Hremove.
    destruct (remove_first_match_data A (vector_get_data A K s c) pred) eqn:Hmatch.
    - destruct p as [found rest].
      inversion Hremove; subst.
      eapply remove_first_match_data_pred. exact Hmatch.
    - discriminate.
  Qed.

  (** Snapshot preserves data *)
  Theorem vector_snapshot_preserves :
    forall (s : vector_storage A K) (c : nat),
      vector_get_data A K (vector_snapshot A K s) c =
      vector_get_data A K s c.
  Proof.
    intros s c.
    unfold vector_snapshot, vector_get_data.
    reflexivity.
  Qed.

  (** Vector never runs out of channels (unlike fixed array) *)
  Theorem vector_gensym_always_succeeds :
    forall (s : vector_storage A K),
      exists c s', vector_gensym A K s = (c, s').
  Proof.
    intros s.
    unfold vector_gensym.
    eexists. eexists. reflexivity.
  Qed.

End VectorStoreProperties.

(** ** Comparison with Fixed Array *)

(** Key difference: Vector gensym always succeeds (see vector_gensym_always_succeeds),
    while FixedArray can fail when capacity is exhausted.

    The infinite capacity property follows from:
    1. vector_gensym_always_succeeds - gensym always returns a fresh channel
    2. vector_gensym_monotonic - sequential gensyms are ordered
    3. vector_gensym_distinct - sequential gensyms are distinct

    This means we can always generate more channels, unlike fixed-size stores. *)
