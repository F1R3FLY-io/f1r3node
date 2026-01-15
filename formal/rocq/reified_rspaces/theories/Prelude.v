(** * Prelude: Core Definitions for Reified RSpaces

    This module provides foundational definitions used throughout
    the Reified RSpaces proof development.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/types.rs
*)

From Coq Require Import String List ZArith Bool Lia.
From Coq Require Import Classes.DecidableClass.
Import ListNotations.

(** ** Space Identifiers *)

(** A SpaceId uniquely identifies a space instance. *)
Definition SpaceId := list Z.

(** The default/root space has all zeros. *)
Definition default_space_id : SpaceId := repeat 0%Z 32.

(** SpaceId equality is decidable. *)
Lemma space_id_eq_dec : forall (s1 s2 : SpaceId), {s1 = s2} + {s1 <> s2}.
Proof.
  decide equality.
  apply Z.eq_dec.
Defined.

(** ** Channel Qualifiers *)

(** Qualifiers control persistence and concurrency behavior. *)
Inductive ChannelQualifier :=
  | QDefault  (* Persistent, concurrent access *)
  | QTemp     (* Non-persistent, concurrent access *)
  | QSeq.     (* Non-persistent, sequential, restricted *)

(** Qualifier equality is decidable. *)
Lemma qualifier_eq_dec : forall (q1 q2 : ChannelQualifier),
  {q1 = q2} + {q1 <> q2}.
Proof.
  decide equality.
Defined.

(** ** Inner Collection Types *)

(** Types of collections for storing data at channels. *)
Inductive InnerCollectionType :=
  | ICBag           (* Multiset, any element can match *)
  | ICQueue         (* FIFO, only head matches *)
  | ICStack         (* LIFO, only top matches *)
  | ICSet           (* Unique elements, idempotent sends *)
  | ICCell          (* At most one element *)
  | ICPriorityQueue (priorities : nat)  (* Priority levels *)
  | ICVectorDB (dimensions : nat).      (* Similarity-based matching *)

(** ** Outer Storage Types *)

(** Types of storage for indexing channels. *)
Inductive OuterStorageType :=
  | OSHashMap                           (* O(1) lookup *)
  | OSPathMap                           (* Hierarchical with prefix matching *)
  | OSArray (max_size : nat) (cyclic : bool)  (* Fixed size *)
  | OSVector                            (* Unbounded *)
  | OSHashSet.                          (* Presence-only *)

(** ** Space Configuration *)

Record SpaceConfig := mkSpaceConfig {
  config_qualifier : ChannelQualifier;
  config_inner : InnerCollectionType;
  config_outer : OuterStorageType;
}.

(** Default configuration: HashMap + Bag + Default qualifier *)
Definition default_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICBag OSHashMap.

(** PathMap configuration (recommended default per spec) *)
Definition pathmap_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICBag OSPathMap.

(** Temp space configuration *)
Definition temp_space_config : SpaceConfig :=
  mkSpaceConfig QTemp ICBag OSHashMap.

(** Seq space configuration *)
Definition seq_space_config : SpaceConfig :=
  mkSpaceConfig QSeq ICBag OSHashSet.

(** Queue space configuration *)
Definition queue_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICQueue OSHashMap.

(** Stack space configuration *)
Definition stack_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICStack OSHashMap.

(** ** Channel Ownership *)

(** A channel can be owned by a specific space or be unassigned. *)
Definition ChannelOwnership (C : Type) := C -> option SpaceId.

(** ** Use Block Stack *)

(** Use blocks create a stack of default spaces per process. *)
Definition UseBlockStack := list SpaceId.

(** Push a space onto the use block stack. *)
Definition push_use_block (stack : UseBlockStack) (s : SpaceId) : UseBlockStack :=
  s :: stack.

(** Pop a space from the use block stack. *)
Definition pop_use_block (stack : UseBlockStack) : option (SpaceId * UseBlockStack) :=
  match stack with
  | [] => None
  | s :: rest => Some (s, rest)
  end.

(** Get the effective default space (top of stack or global default). *)
Definition effective_default (stack : UseBlockStack) (global : option SpaceId)
  : option SpaceId :=
  match stack with
  | [] => global
  | s :: _ => Some s
  end.

(** ** Key Properties *)

(** Seq channels cannot be sent to other processes. *)
Definition is_mobile (q : ChannelQualifier) : bool :=
  match q with
  | QSeq => false
  | _ => true
  end.

(** Temp spaces are cleared on checkpoint. *)
Definition is_persistent (q : ChannelQualifier) : bool :=
  match q with
  | QDefault => true
  | _ => false
  end.

(** ** Theorems about Core Types *)

(** The default space ID is unique (all zeros).
    This theorem characterizes the default space ID. *)
Theorem default_space_unique : forall s : SpaceId,
  s = default_space_id <-> (forall n, nth n s 0%Z = 0%Z) /\ length s = 32.
Proof.
  intros s.
  split.
  - (* Forward: s = default_space_id -> ... *)
    intros Heq.
    subst s.
    unfold default_space_id.
    split.
    + (* forall n, nth n (repeat 0 32) 0 = 0 *)
      intros n.
      destruct (Nat.lt_ge_cases n 32) as [Hlt | Hge].
      * (* n < 32: use nth_repeat *)
        now rewrite nth_repeat.
      * (* n >= 32: use nth_overflow *)
        rewrite nth_overflow.
        -- reflexivity.
        -- rewrite repeat_length. lia.
    + (* length (repeat 0 32) = 32 *)
      apply repeat_length.
  - (* Backward: ... -> s = default_space_id *)
    intros [Hnth Hlen].
    unfold default_space_id.
    apply nth_ext with (d := 0%Z) (d' := 0%Z).
    + (* length s = length (repeat 0 32) *)
      rewrite repeat_length. exact Hlen.
    + (* forall n, n < length s -> nth n s 0 = nth n (repeat 0 32) 0 *)
      intros n Hn.
      rewrite Hnth.
      rewrite Hlen in Hn.
      now rewrite nth_repeat.
Qed.

(** Qualifier properties are mutually exclusive for mobility. *)
Theorem mobile_not_seq : forall q,
  is_mobile q = true <-> q <> QSeq.
Proof.
  intros q.
  split.
  - intros H.
    destruct q; simpl in H.
    + intro contra. discriminate contra.
    + intro contra. discriminate contra.
    + discriminate H.
  - intros H.
    destruct q; simpl; try reflexivity.
    exfalso. apply H. reflexivity.
Qed.

(** Persistence implies mobility (Default is both). *)
Theorem persistent_implies_mobile : forall q,
  is_persistent q = true -> is_mobile q = true.
Proof.
  intros q H.
  destruct q; simpl in *; auto; discriminate.
Qed.

(** ** Notations *)

Notation "⊥" := None (at level 0).
Notation "⟨ x ⟩" := (Some x) (at level 0, format "⟨ x ⟩").
