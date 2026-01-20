(** * SpaceFactory Specification

    This module specifies the SpaceFactory pattern for creating
    properly configured RSpace instances. The factory ensures that
    created spaces satisfy all invariants from construction.

    Reference: Spec "Reifying RSpaces.md" lines 720-812
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude Match GenericRSpace.
Import ListNotations.

(** ** Factory Configuration *)

Section FactoryConfig.

  (** Inner collection type selection *)
  Inductive FactoryInnerType :=
    | FI_Bag            (* Unordered multiset *)
    | FI_Queue          (* FIFO ordering *)
    | FI_Stack          (* LIFO ordering *)
    | FI_Set            (* Unique elements *)
    | FI_Cell           (* At most one element *)
    | FI_PriorityQueue  (* Priority levels *)
    | FI_VectorDB.      (* Similarity matching *)

  (** Outer storage type selection *)
  Inductive FactoryOuterType :=
    | FO_HashMap        (* O(1) lookup *)
    | FO_PathMap        (* Hierarchical with prefix matching *)
    | FO_FixedArray     (* Fixed size array *)
    | FO_CyclicArray    (* Wraparound array *)
    | FO_Vector         (* Growable array *)
    | FO_HashSet.       (* Presence-only storage *)

  (** Factory configuration record *)
  Record FactoryConfig := mkFactoryConfig {
    fc_qualifier : SpaceQualifier;    (* Default=0, Temp=1, Seq=2 *)
    fc_inner_type : FactoryInnerType;
    fc_outer_type : FactoryOuterType;
    fc_max_size : option nat;         (* For fixed/cyclic arrays *)
  }.

  (** Qualifier constants from GenericRSpace *)
  Definition FC_QUALIFIER_DEFAULT : SpaceQualifier := 0.
  Definition FC_QUALIFIER_TEMP : SpaceQualifier := 1.
  Definition FC_QUALIFIER_SEQ : SpaceQualifier := 2.

End FactoryConfig.

(** ** Configuration Validation *)

Section ConfigValidation.

  (** Seq spaces cannot use HashSet (concurrency issues) *)
  Definition seq_compatible_outer (outer : FactoryOuterType) : bool :=
    match outer with
    | FO_HashSet => false
    | _ => true
    end.

  (** Fixed/Cyclic arrays require max_size *)
  Definition array_size_valid (outer : FactoryOuterType) (max_size : option nat) : bool :=
    match outer with
    | FO_FixedArray | FO_CyclicArray =>
      match max_size with
      | Some n => (0 <? n)%nat
      | None => false
      end
    | _ => true
    end.

  (** VectorDB requires specific outer types *)
  Definition vectordb_compatible_outer (inner : FactoryInnerType) (outer : FactoryOuterType) : bool :=
    match inner with
    | FI_VectorDB =>
      match outer with
      | FO_HashMap | FO_Vector => true
      | _ => false
      end
    | _ => true
    end.

  (** Complete configuration validation *)
  Definition valid_config (cfg : FactoryConfig) : bool :=
    let qual := fc_qualifier cfg in
    let inner := fc_inner_type cfg in
    let outer := fc_outer_type cfg in
    let max_size := fc_max_size cfg in
    (* Seq qualifier restrictions *)
    (if Nat.eqb qual FC_QUALIFIER_SEQ then seq_compatible_outer outer else true) &&
    (* Array size requirements *)
    array_size_valid outer max_size &&
    (* VectorDB compatibility *)
    vectordb_compatible_outer inner outer.

End ConfigValidation.

(** ** Factory Trait *)

Section SpaceFactoryTrait.
  Variable A K : Type.

  (** Factory trait abstraction *)
  Class SpaceFactory := {
    (** Create a new space from configuration *)
    sf_create : FactoryConfig -> option (GenericRSpaceState A K);

    (** Validate configuration before creation *)
    sf_validate : FactoryConfig -> bool;

    (** Validation is consistent with valid_config *)
    sf_validate_correct : forall cfg,
      sf_validate cfg = valid_config cfg;

    (** Creation succeeds for valid configs *)
    sf_create_valid : forall cfg,
      sf_validate cfg = true ->
      exists state, sf_create cfg = Some state;

    (** Creation fails for invalid configs *)
    sf_create_invalid : forall cfg,
      sf_validate cfg = false ->
      sf_create cfg = None;

    (** Factory must respect qualifier in created states *)
    sf_respects_qualifier : forall cfg state,
      sf_create cfg = Some state ->
      gr_qualifier state = fc_qualifier cfg;

    (** Factory must produce empty states *)
    sf_produces_empty : forall cfg state,
      sf_create cfg = Some state ->
      gr_data_store state = @empty_data_store A /\
      gr_cont_store state = @empty_cont_store A K /\
      gr_joins state = empty_joins;
  }.

End SpaceFactoryTrait.

(** ** Factory Correctness *)

Section FactoryCorrectness.
  Variable A K : Type.
  Variable M : Match A A.
  Context `{SF : SpaceFactory A K}.

  (** Created state has correct qualifier *)
  Theorem factory_respects_qualifier :
    forall cfg state,
      @sf_create A K SF cfg = Some state ->
      gr_qualifier state = fc_qualifier cfg.
  Proof.
    intros cfg state Hcreate.
    apply (@sf_respects_qualifier A K SF cfg state Hcreate).
  Qed.

  (** Created state satisfies basic invariant (empty = no pending matches) *)
  Theorem factory_produces_valid_state :
    forall cfg state,
      @sf_validate A K SF cfg = true ->
      @sf_create A K SF cfg = Some state ->
      gr_data_store state = @empty_data_store A /\
      gr_cont_store state = @empty_cont_store A K /\
      gr_joins state = empty_joins.
  Proof.
    intros cfg state Hvalid Hcreate.
    apply (@sf_produces_empty A K SF cfg state Hcreate).
  Qed.

  (** Empty state trivially satisfies no_pending_match *)
  Theorem empty_state_no_pending_match :
    forall qualifier,
      no_pending_match A K M (empty_gr_state qualifier).
  Proof.
    intros qualifier.
    unfold no_pending_match.
    intros ch data persist Hdata.
    (* Empty state has no data *)
    simpl in Hdata.
    destruct Hdata.
  Qed.

  (** Factory creates empty state (no data, no continuations) *)
  Theorem factory_creates_empty :
    forall cfg state,
      @sf_validate A K SF cfg = true ->
      @sf_create A K SF cfg = Some state ->
      no_pending_match A K M state.
  Proof.
    intros cfg state Hvalid Hcreate.
    (* From factory_produces_valid_state, state stores are empty *)
    destruct (factory_produces_valid_state cfg state Hvalid Hcreate)
      as [Hdata [Hcont Hjoins]].
    unfold no_pending_match.
    intros ch data persist Hdata_in.
    rewrite Hdata in Hdata_in.
    unfold empty_data_store in Hdata_in.
    destruct Hdata_in.
  Qed.

End FactoryCorrectness.

(** ** Default Factory Instance *)

Section DefaultFactory.
  Variable A K : Type.

  (** Simple factory that creates empty states *)
  Definition default_factory_create (cfg : FactoryConfig) : option (GenericRSpaceState A K) :=
    if valid_config cfg
    then Some (empty_gr_state (fc_qualifier cfg))
    else None.

  (** Prove that valid_config satisfies validate_correct *)
  Lemma valid_config_correct : forall cfg, valid_config cfg = valid_config cfg.
  Proof. reflexivity. Qed.

  (** Prove that creation succeeds for valid configs *)
  Lemma default_create_valid : forall cfg,
    valid_config cfg = true ->
    exists state, default_factory_create cfg = Some state.
  Proof.
    intros cfg Hvalid.
    unfold default_factory_create.
    rewrite Hvalid.
    eexists. reflexivity.
  Qed.

  (** Prove that creation fails for invalid configs *)
  Lemma default_create_invalid : forall cfg,
    valid_config cfg = false ->
    default_factory_create cfg = None.
  Proof.
    intros cfg Hinvalid.
    unfold default_factory_create.
    rewrite Hinvalid. reflexivity.
  Qed.

  (** Prove that default factory respects qualifier *)
  Lemma default_respects_qualifier : forall cfg state,
    default_factory_create cfg = Some state ->
    gr_qualifier state = fc_qualifier cfg.
  Proof.
    intros cfg state Hcreate.
    unfold default_factory_create in Hcreate.
    destruct (valid_config cfg) eqn:Hvalid.
    - injection Hcreate as Hstate.
      subst state. simpl. reflexivity.
    - discriminate Hcreate.
  Qed.

  (** Prove that default factory creates empty state *)
  Lemma default_produces_empty : forall cfg state,
    default_factory_create cfg = Some state ->
    gr_data_store state = @empty_data_store A /\
    gr_cont_store state = @empty_cont_store A K /\
    gr_joins state = empty_joins.
  Proof.
    intros cfg state Hcreate.
    unfold default_factory_create in Hcreate.
    destruct (valid_config cfg) eqn:Hvalid.
    - injection Hcreate as Hstate.
      subst state. simpl.
      repeat split; reflexivity.
    - discriminate Hcreate.
  Qed.

  (** Default factory implementation *)
  Instance DefaultSpaceFactory : SpaceFactory A K := {
    sf_create := default_factory_create;
    sf_validate := valid_config;
    sf_validate_correct := valid_config_correct;
    sf_create_valid := default_create_valid;
    sf_create_invalid := default_create_invalid;
    sf_respects_qualifier := default_respects_qualifier;
    sf_produces_empty := default_produces_empty;
  }.

  (** Default factory respects qualifier (legacy theorem name) *)
  Theorem default_factory_qualifier :
    forall cfg state,
      default_factory_create cfg = Some state ->
      gr_qualifier state = fc_qualifier cfg.
  Proof.
    exact default_respects_qualifier.
  Qed.

  (** Default factory creates empty state (legacy theorem name) *)
  Theorem default_factory_empty :
    forall cfg state,
      default_factory_create cfg = Some state ->
      gr_data_store state = @empty_data_store A /\
      gr_cont_store state = @empty_cont_store A K /\
      gr_joins state = empty_joins.
  Proof.
    exact default_produces_empty.
  Qed.

End DefaultFactory.

(** ** Configuration Validation Properties *)

Section ValidationProperties.

  (** Valid default configuration *)
  Theorem default_config_valid :
    valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_HashMap None) = true.
  Proof.
    reflexivity.
  Qed.

  (** Valid temp configuration *)
  Theorem temp_config_valid :
    valid_config (mkFactoryConfig FC_QUALIFIER_TEMP FI_Bag FO_HashMap None) = true.
  Proof.
    reflexivity.
  Qed.

  (** Seq with HashSet is invalid *)
  Theorem seq_hashset_invalid :
    valid_config (mkFactoryConfig FC_QUALIFIER_SEQ FI_Bag FO_HashSet None) = false.
  Proof.
    reflexivity.
  Qed.

  (** Fixed array without size is invalid *)
  Theorem fixed_array_no_size_invalid :
    valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_FixedArray None) = false.
  Proof.
    reflexivity.
  Qed.

  (** Fixed array with size is valid *)
  Theorem fixed_array_with_size_valid :
    valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_FixedArray (Some 100)) = true.
  Proof.
    reflexivity.
  Qed.

  (** VectorDB with HashMap is valid *)
  Theorem vectordb_hashmap_valid :
    valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_VectorDB FO_HashMap None) = true.
  Proof.
    reflexivity.
  Qed.

  (** VectorDB with PathMap is invalid *)
  Theorem vectordb_pathmap_invalid :
    valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_VectorDB FO_PathMap None) = false.
  Proof.
    reflexivity.
  Qed.

End ValidationProperties.

(** ** Factory Composition *)

Section FactoryComposition.
  Variable A K : Type.
  Variable M : Match A A.

  (** Create multiple spaces with same configuration *)
  Definition create_many (factory : FactoryConfig -> option (GenericRSpaceState A K))
                         (cfg : FactoryConfig) (n : nat)
    : list (GenericRSpaceState A K) :=
    match factory cfg with
    | None => []
    | Some state =>
      (* Create n copies (each would have different name counter in practice) *)
      repeat state n
    end.

  (** All created spaces satisfy invariant *)
  Theorem create_many_all_valid :
    forall cfg n state,
      valid_config cfg = true ->
      In state (create_many (default_factory_create A K) cfg n) ->
      no_pending_match A K M state.
  Proof.
    intros cfg n state Hvalid Hin.
    unfold create_many in Hin.
    unfold default_factory_create in Hin.
    rewrite Hvalid in Hin.
    apply repeat_spec in Hin.
    subst state.
    apply (GenericRSpace.empty_state_no_pending_match A K M).
  Qed.

End FactoryComposition.
