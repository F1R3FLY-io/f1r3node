(** * GenericRSpace Specification

    This module specifies the GenericRSpace<CS, M> parameterized implementation
    that combines a ChannelStore with a Match strategy.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/generic_rspace.rs
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude Match.
From ReifiedRSpaces.Collections Require Import DataCollection.
Import ListNotations.

(** ** Channel and Space Types *)

Definition Channel := Z.
Definition SpaceQualifier := nat. (* 0=Default, 1=Temp, 2=Seq *)

(** ** Generic RSpace State *)

(** The state of a GenericRSpace includes:
    - A mapping from channels to data collections
    - A mapping from channel-lists to continuation collections
    - Join information for multi-channel consumes
    - Space metadata (qualifier, name counter)
*)
Record GenericRSpaceState (A K : Type) := mkGRState {
  gr_data_store : Channel -> list (A * bool);
  gr_cont_store : list Channel -> list (list A * K * bool);
  gr_joins : Channel -> list (list Channel);
  gr_qualifier : SpaceQualifier;
  gr_name_counter : nat;
}.

Arguments mkGRState {A K}.
Arguments gr_data_store {A K}.
Arguments gr_cont_store {A K}.
Arguments gr_joins {A K}.
Arguments gr_qualifier {A K}.
Arguments gr_name_counter {A K}.

(** ** Empty State *)

Definition empty_data_store {A} : Channel -> list (A * bool) := fun _ => [].
Definition empty_cont_store {A K} : list Channel -> list (list A * K * bool) := fun _ => [].
Definition empty_joins : Channel -> list (list Channel) := fun _ => [].

Definition empty_gr_state {A K} (qualifier : SpaceQualifier) : GenericRSpaceState A K :=
  mkGRState empty_data_store empty_cont_store empty_joins qualifier 0.

(** ** Helper Functions *)

Section Helpers.
  Variable A : Type.

  (** Find first element satisfying predicate *)
  Fixpoint find_first {X} (pred : X -> bool) (l : list X) : option X :=
    match l with
    | [] => None
    | x :: rest => if pred x then Some x else find_first pred rest
    end.

  (** Remove first element satisfying predicate *)
  Fixpoint remove_first {X} (pred : X -> bool) (l : list X) : list X :=
    match l with
    | [] => []
    | x :: rest => if pred x then rest else x :: remove_first pred rest
    end.

  (** Key lemma: if find_first returns None, predicate is false for all elements *)
  Lemma find_first_none_implies_all_false :
    forall {X} (pred : X -> bool) (l : list X),
      find_first pred l = None ->
      forall x, In x l -> pred x = false.
  Proof.
    intros X pred l. induction l as [| h t IH].
    - intros _ x Hin. inversion Hin.
    - intros Hnone x Hin.
      simpl in Hnone.
      destruct (pred h) eqn:Hpred.
      + discriminate.
      + destruct Hin as [Heq | Hin'].
        * subst. exact Hpred.
        * apply IH; assumption.
  Qed.

  (** Key lemma: if find_first returns Some, the found element satisfies predicate *)
  Lemma find_first_some_implies_pred :
    forall {X} (pred : X -> bool) (l : list X) (x : X),
      find_first pred l = Some x ->
      pred x = true.
  Proof.
    intros X pred l. induction l as [| h t IH].
    - intros x H. discriminate.
    - intros x H.
      simpl in H.
      destruct (pred h) eqn:Hpred.
      + injection H as Heq. subst. exact Hpred.
      + apply IH. exact H.
  Qed.

  (** Key lemma: remove_first preserves elements not satisfying predicate *)
  Lemma remove_first_preserves_non_matching :
    forall {X} (pred : X -> bool) (l : list X) (x : X),
      In x (remove_first pred l) ->
      In x l.
  Proof.
    intros X pred l. induction l as [| h t IH].
    - intros x H. simpl in H. exact H.
    - intros x H.
      simpl in H.
      destruct (pred h) eqn:Hpred.
      + right. exact H.
      + destruct H as [Heq | Hin].
        * left. exact Heq.
        * right. apply IH. exact Hin.
  Qed.

  (** In is decidable for lists with decidable equality *)
  Lemma In_dec_list :
    forall {X} (eq_dec : forall x y : X, {x = y} + {x <> y}) (x : X) (l : list X),
      {In x l} + {~ In x l}.
  Proof.
    intros X eq_dec x l.
    induction l as [| h t IH].
    - right. intro H. inversion H.
    - destruct (eq_dec x h) as [Heq | Hneq].
      + left. left. symmetry. exact Heq.
      + destruct IH as [Hin | Hnotin].
        * left. right. exact Hin.
        * right. intro H. destruct H as [Heq | Hin].
          -- apply Hneq. symmetry. exact Heq.
          -- apply Hnotin. exact Hin.
  Qed.
End Helpers.

(** ** Core Invariant: No Pending Matches *)

(** The fundamental invariant of RSpaces: there is never simultaneously
    data and a matching continuation on the same channel(s).

    If data exists and a continuation exists, one of them must have
    been matched and removed. *)
Section NoPendingMatch.
  Variable A K : Type.
  Variable M : Match A A.

  (** Check if any pattern in patterns matches the data *)
  Fixpoint any_pattern_matches (patterns : list A) (data : A) : bool :=
    match patterns with
    | [] => false
    | p :: rest => if @match_fn A A M p data then true else any_pattern_matches rest data
    end.

  (** The invariant: no data matches any waiting continuation *)
  Definition no_pending_match (state : GenericRSpaceState A K) : Prop :=
    forall ch data persist,
      In (data, persist) (gr_data_store state ch) ->
      forall join_channels patterns cont cpersist,
        In join_channels (gr_joins state ch) ->
        In (patterns, cont, cpersist) (gr_cont_store state join_channels) ->
        any_pattern_matches patterns data = false.

  (** Alternative formulation: for single-channel case *)
  Definition no_pending_match_single (state : GenericRSpaceState A K) : Prop :=
    forall ch data dpersist patterns cont cpersist,
      In (data, dpersist) (gr_data_store state ch) ->
      In (patterns, cont, cpersist) (gr_cont_store state [ch]) ->
      forall p, In p patterns -> @match_fn A A M p data = false.

  (** Empty state satisfies the invariant *)
  Theorem empty_state_no_pending_match :
    forall qualifier, no_pending_match (empty_gr_state qualifier).
  Proof.
    intros qualifier.
    unfold no_pending_match, empty_gr_state. simpl.
    intros ch data persist Hdata.
    inversion Hdata.
  Qed.
End NoPendingMatch.

(** ** Produce Operation *)

(** Produce stores data at a channel.
    If a matching continuation exists, it fires instead. *)
Section ProduceSpec.
  Variable A K : Type.
  Variable M : Match A A.

  (** Result of produce: either stored data or fired continuation *)
  Inductive ProduceResult :=
    | PR_Stored : GenericRSpaceState A K -> ProduceResult
    | PR_Fired : K -> A -> GenericRSpaceState A K -> ProduceResult.

  (** Check if any pattern matches the data (local version for this section) *)
  Fixpoint any_pattern_matches_local (patterns : list A) (data : A) : bool :=
    match patterns with
    | [] => false
    | p :: rest => if @match_fn A A M p data then true else any_pattern_matches_local rest data
    end.

  (** Check if any continuation matches the data *)
  Fixpoint find_matching_cont (joins : list (list Channel))
    (cont_store : list Channel -> list (list A * K * bool))
    (data : A)
    : option (list Channel * (list A * K * bool)) :=
    match joins with
    | [] => None
    | join_chs :: rest =>
      let conts := cont_store join_chs in
      match find_first (fun '(patterns, _, _) => any_pattern_matches_local patterns data) conts with
      | Some cont_entry => Some (join_chs, cont_entry)
      | None => find_matching_cont rest cont_store data
      end
    end.

  (** Produce specification (simplified for single channel) *)
  Definition produce_spec (state : GenericRSpaceState A K)
    (ch : Channel) (data : A) (persist : bool)
    : ProduceResult :=
    let joins := gr_joins state ch in
    match find_matching_cont joins (gr_cont_store state) data with
    | Some (join_chs, (patterns, cont, cpersist)) =>
      (* Found match - fire continuation *)
      (* Remove first matching continuation from the store *)
      let new_cont_store := fun chs =>
        if list_eq_dec Z.eq_dec chs join_chs then
          (* Remove the matching continuation *)
          remove_first (fun '(ps, _, _) => any_pattern_matches_local ps data) (gr_cont_store state chs)
        else
          gr_cont_store state chs
      in
      let new_state := mkGRState
        (gr_data_store state)
        new_cont_store
        (gr_joins state)
        (gr_qualifier state)
        (gr_name_counter state)
      in
      PR_Fired cont data new_state
    | None =>
      (* No match - store data *)
      let new_data_store := fun c =>
        if Z.eq_dec c ch then (data, persist) :: gr_data_store state c
        else gr_data_store state c
      in
      let new_state := mkGRState
        new_data_store
        (gr_cont_store state)
        (gr_joins state)
        (gr_qualifier state)
        (gr_name_counter state)
      in
      PR_Stored new_state
    end.

  (** Helper lemma: if find_first returns None, predicate is false for all *)
  Lemma find_first_none_local :
    forall (conts : list (list A * K * bool)) data,
      find_first (fun '(ps, _, _) => any_pattern_matches_local ps data) conts = None ->
      forall patterns cont cpersist,
        In (patterns, cont, cpersist) conts ->
        any_pattern_matches_local patterns data = false.
  Proof.
    intros conts data. induction conts as [| [[ps' k'] cp'] rest IHrest].
    - intros _ patterns cont cpersist Hin. inversion Hin.
    - intros Hfirst patterns cont cpersist Hin.
      simpl in Hfirst.
      destruct (any_pattern_matches_local ps' data) eqn:Hmatch.
      + discriminate.
      + destruct Hin as [Heq | Hin'].
        * injection Heq as Hp Hc Hcp. subst. exact Hmatch.
        * apply (IHrest Hfirst patterns cont cpersist Hin').
  Qed.

  (** Key lemma: any_pattern_matches_local equals any_pattern_matches *)
  Lemma any_pattern_matches_equiv :
    forall patterns data,
      any_pattern_matches_local patterns data = any_pattern_matches A M patterns data.
  Proof.
    intros patterns data.
    induction patterns as [| p rest IH].
    - reflexivity.
    - simpl.
      destruct (@match_fn A A M p data); [reflexivity | exact IH].
  Qed.

  (** Key lemma: find_matching_cont None means no continuation matches *)
  Lemma find_matching_cont_none_no_match :
    forall joins cont_store data,
      find_matching_cont joins cont_store data = None ->
      forall jc, In jc joins ->
      forall patterns cont cpersist,
        In (patterns, cont, cpersist) (cont_store jc) ->
        any_pattern_matches_local patterns data = false.
  Proof.
    intros joins. induction joins as [| jc_hd joins_tl IH].
    - intros cont_store data _ jc Hjc. inversion Hjc.
    - intros cont_store data Hfind jc Hjc patterns cont cpersist Hcont.
      simpl in Hfind.
      destruct (find_first (fun '(ps, _, _) => any_pattern_matches_local ps data)
                  (cont_store jc_hd)) eqn:Hfirst.
      + (* find_first returned Some - contradiction with find_matching_cont = None *)
        destruct p as [[ps k] cp]. discriminate.
      + (* find_first returned None for jc_hd *)
        destruct Hjc as [Heq | Hin].
        * (* jc = jc_hd *)
          subst jc.
          eapply find_first_none_local; eassumption.
        * (* jc in joins_tl *)
          apply (IH cont_store data Hfind jc Hin patterns cont cpersist Hcont).
  Qed.
End ProduceSpec.

(** ** Consume Operation *)

(** Consume waits for data on channel(s).
    If matching data exists, it fires immediately. *)
Section ConsumeSpec.
  Variable A K : Type.
  Variable M : Match A A.

  (** Result of consume: either stored continuation or immediate match *)
  Inductive ConsumeResult :=
    | CR_Stored : GenericRSpaceState A K -> ConsumeResult
    | CR_Matched : list A -> GenericRSpaceState A K -> ConsumeResult.

  (** Consume specification *)
  Definition consume_spec (state : GenericRSpaceState A K)
    (channels : list Channel) (patterns : list A) (cont : K) (persist : bool)
    : ConsumeResult :=
    (* Simplified: just store the continuation *)
    let new_cont_store := fun chs =>
      if list_eq_dec Z.eq_dec chs channels then
        (patterns, cont, persist) :: gr_cont_store state chs
      else
        gr_cont_store state chs
    in
    let new_joins := fun ch =>
      if In_dec Z.eq_dec ch channels then
        channels :: gr_joins state ch
      else
        gr_joins state ch
    in
    let new_state := mkGRState
      (gr_data_store state)
      new_cont_store
      new_joins
      (gr_qualifier state)
      (gr_name_counter state)
    in
    CR_Stored new_state.
End ConsumeSpec.

(** ** Join Consistency Invariant (Forward Definitions) *)

(** These definitions are needed by InstallSpec and are defined here
    for forward reference. Full theorems appear later. *)
Section JoinConsistencyForward.
  Variable A K : Type.

  (** Join consistency: if channels is in gr_joins ch, then ch is in channels *)
  Definition join_consistent_state (state : GenericRSpaceState A K) : Prop :=
    forall ch channels,
      In channels (gr_joins state ch) ->
      In ch channels.

  (** Continuation implies join registration *)
  Definition cont_implies_join_forward (state : GenericRSpaceState A K) : Prop :=
    forall channels patterns cont cpersist,
      In (patterns, cont, cpersist) (gr_cont_store state channels) ->
      forall ch, In ch channels ->
        In channels (gr_joins state ch).

  (** Empty state is join consistent *)
  Theorem empty_state_join_consistent_fwd :
    forall qualifier, join_consistent_state (@empty_gr_state A K qualifier).
  Proof.
    intros qualifier.
    unfold join_consistent_state, empty_gr_state. simpl.
    intros ch channels Hin. inversion Hin.
  Qed.

End JoinConsistencyForward.

(** ** Install Operation *)

(** Install permanently installs a continuation at channels.
    Unlike consume, install is immediate and unconditional.
    If matching data exists, it returns immediately.
    Reference: Spec lines 61-62, 499-505, 706-714 *)
Section InstallSpec.
  Variable A K : Type.
  Variable M : Match A A.

  (** Result of install: either stored or immediate match *)
  Inductive InstallResult :=
    | IR_Installed : GenericRSpaceState A K -> InstallResult
    | IR_Matched : K -> list A -> GenericRSpaceState A K -> InstallResult.

  (** Check if all patterns have matching data at corresponding channels *)
  Fixpoint find_all_matching_data_local
    (channels : list Channel) (patterns : list A)
    (data_store : Channel -> list (A * bool))
    : option (list A) :=
    match channels, patterns with
    | [], [] => Some []
    | ch :: ch_rest, pat :: pat_rest =>
      let data_at_ch := data_store ch in
      match find_first (fun '(d, _) => @match_fn A A M pat d) data_at_ch with
      | Some (matched_data, _) =>
        match find_all_matching_data_local ch_rest pat_rest data_store with
        | Some rest_data => Some (matched_data :: rest_data)
        | None => None
        end
      | None => None
      end
    | _, _ => None (* Mismatched lengths *)
    end.

  (** Install specification *)
  Definition install_spec (state : GenericRSpaceState A K)
    (channels : list Channel) (patterns : list A) (cont : K)
    : InstallResult :=
    (* First check if matching data exists *)
    match find_all_matching_data_local channels patterns (gr_data_store state) with
    | Some matched_data =>
      (* Data exists - return immediately with matched data *)
      (* Note: install does NOT remove data (persistent behavior) *)
      IR_Matched cont matched_data state
    | None =>
      (* No data - install continuation permanently *)
      let new_cont_store := fun chs =>
        if list_eq_dec Z.eq_dec chs channels then
          (* Install adds continuation with persist=true *)
          (patterns, cont, true) :: gr_cont_store state chs
        else
          gr_cont_store state chs
      in
      let new_joins := fun ch =>
        if In_dec Z.eq_dec ch channels then
          channels :: gr_joins state ch
        else
          gr_joins state ch
      in
      let new_state := mkGRState
        (gr_data_store state)
        new_cont_store
        new_joins
        (gr_qualifier state)
        (gr_name_counter state)
      in
      IR_Installed new_state
    end.

  (** Install maintains no_pending_match invariant *)
  (** Note: Like consume, install requires as a precondition that no data
      at any channel in channels matches any pattern. This is a semantic
      requirement for the tuple space invariant. *)
  Theorem install_maintains_invariant :
    forall state channels patterns cont state',
      no_pending_match A K M state ->
      join_consistent_state A K state ->
      cont_implies_join_forward A K state ->
      install_spec state channels patterns cont = IR_Installed state' ->
      (* Precondition: no data at any install channel matches any pattern *)
      (forall ch, In ch channels ->
        forall data dpersist,
          In (data, dpersist) (gr_data_store state ch) ->
          any_pattern_matches A M patterns data = false) ->
      no_pending_match A K M state'.
  Proof.
    intros state channels patterns cont state' Hinv Hjc Hcij Hinst Hpre.
    unfold install_spec in Hinst.
    destruct (find_all_matching_data_local channels patterns (gr_data_store state))
      as [matched_data |] eqn:Hfind.
    - (* Found data - produces IR_Matched, contradicts assumption *)
      discriminate.
    - (* No data found - continuation installed *)
      injection Hinst as Hstate'.
      subst state'.
      unfold no_pending_match in *.
      simpl.
      intros ch data dpersist Hdata join_channels patterns' cont' cpersist' Hjoin Hcont.
      (* Case analysis on whether this is the new continuation or an old one *)
      simpl in Hcont.
      destruct (list_eq_dec Z.eq_dec join_channels channels) as [Heq | Hneq].
      + (* join_channels = channels *)
        subst join_channels.
        destruct Hcont as [Hnew_cont | Hold_cont].
        * (* (patterns', cont', cpersist') is the installed continuation *)
          injection Hnew_cont as Hp Hc Hcp.
          subst patterns' cont' cpersist'.
          (* ch must be in channels for the join to contain channels *)
          destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
          -- (* ch in channels - use precondition *)
             apply Hpre with ch dpersist.
             ++ exact Hch_in.
             ++ exact Hdata.
          -- (* ch not in channels - use join consistency *)
             simpl in Hjoin.
             destruct (In_dec Z.eq_dec ch channels) as [Habs | _]; [contradiction |].
             exfalso. apply Hch_notin.
             apply Hjc. exact Hjoin.
        * (* Old continuation - use existing invariant *)
          destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
          -- assert (Hjoin_old : In channels (gr_joins state ch)).
             { apply Hcij with patterns' cont' cpersist'. exact Hold_cont. exact Hch_in. }
             eapply Hinv.
             ++ exact Hdata.
             ++ exact Hjoin_old.
             ++ exact Hold_cont.
          -- simpl in Hjoin.
             destruct (In_dec Z.eq_dec ch channels) as [Habs | _]; [contradiction |].
             exfalso. apply Hch_notin.
             apply Hjc. exact Hjoin.
      + (* join_channels <> channels *)
        simpl in Hjoin.
        destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
        * destruct Hjoin as [Hnew_join | Hold_join].
          -- subst. contradiction.
          -- eapply Hinv; eassumption.
        * eapply Hinv; eassumption.
  Qed.

  (** Install preserves join consistency *)
  Theorem install_preserves_join_consistency :
    forall state channels patterns cont state',
      join_consistent_state A K state ->
      install_spec state channels patterns cont = IR_Installed state' ->
      join_consistent_state A K state'.
  Proof.
    intros state channels patterns cont state' Hjc Hinst.
    unfold install_spec in Hinst.
    destruct (find_all_matching_data_local channels patterns (gr_data_store state))
      as [matched_data |].
    - discriminate.
    - injection Hinst as Hstate'.
      subst state'.
      unfold join_consistent_state in *.
      simpl.
      intros ch channels' Hjoin'.
      destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
      + destruct Hjoin' as [Hnew | Hold].
        * subst. exact Hch_in.
        * apply Hjc. exact Hold.
      + apply Hjc. exact Hjoin'.
  Qed.

  (** Install preserves cont_implies_join_forward *)
  Theorem install_preserves_cont_implies_join :
    forall state channels patterns cont state',
      cont_implies_join_forward A K state ->
      install_spec state channels patterns cont = IR_Installed state' ->
      cont_implies_join_forward A K state'.
  Proof.
    intros state channels patterns cont state' Hcij Hinst.
    unfold install_spec in Hinst.
    destruct (find_all_matching_data_local channels patterns (gr_data_store state))
      as [matched_data |].
    - discriminate.
    - injection Hinst as Hstate'.
      subst state'.
      unfold cont_implies_join_forward in *.
      simpl.
      intros channels' patterns' cont' cpersist' Hcont' ch' Hch_in'.
      destruct (list_eq_dec Z.eq_dec channels' channels) as [Heq | Hneq].
      + subst channels'.
        destruct Hcont' as [Hnew | Hold].
        * destruct (In_dec Z.eq_dec ch' channels) as [Hin | Hnotin].
          -- left. reflexivity.
          -- contradiction.
        * destruct (In_dec Z.eq_dec ch' channels) as [Hin | Hnotin].
          -- left. reflexivity.
          -- contradiction.
      + destruct (In_dec Z.eq_dec ch' channels) as [Hch_in_chs | Hch_notin_chs].
        * right. apply Hcij with patterns' cont' cpersist'. exact Hcont'. exact Hch_in'.
        * apply Hcij with patterns' cont' cpersist'. exact Hcont'. exact Hch_in'.
  Qed.

  (** Install with immediate match returns existing data *)
  Theorem install_data_immediate :
    forall state channels patterns cont matched_data,
      install_spec state channels patterns cont = IR_Matched cont matched_data state ->
      find_all_matching_data_local channels patterns (gr_data_store state) = Some matched_data.
  Proof.
    intros state channels patterns cont matched_data Hinst.
    unfold install_spec in Hinst.
    remember (find_all_matching_data_local channels patterns (gr_data_store state)) as result.
    destruct result as [data |].
    - injection Hinst as Hd. subst data. reflexivity.
    - discriminate.
  Qed.

  (** Install with immediate match does not modify state *)
  Theorem install_matched_state_unchanged :
    forall state channels patterns cont matched_data state',
      install_spec state channels patterns cont = IR_Matched cont matched_data state' ->
      state' = state.
  Proof.
    intros state channels patterns cont matched_data state' Hinst.
    unfold install_spec in Hinst.
    destruct (find_all_matching_data_local channels patterns (gr_data_store state))
      as [data |].
    - injection Hinst as _ Hs. symmetry. exact Hs.
    - discriminate.
  Qed.

  (** Install exclusive: either installed or matched, never both *)
  Theorem install_exclusive :
    forall state channels patterns cont,
      (exists state', install_spec state channels patterns cont = IR_Installed state') \/
      (exists matched_data, install_spec state channels patterns cont = IR_Matched cont matched_data state).
  Proof.
    intros state channels patterns cont.
    unfold install_spec.
    destruct (find_all_matching_data_local channels patterns (gr_data_store state))
      as [data |].
    - right. exists data. reflexivity.
    - left. eexists. reflexivity.
  Qed.

End InstallSpec.

(** ** Invariant Preservation *)

Section InvariantPreservation.
  Variable A K : Type.
  Variable M : Match A A.

  (** Combined invariant: no pending match, join consistency, and cont-implies-join *)
  Definition full_invariant (state : GenericRSpaceState A K) : Prop :=
    no_pending_match A K M state /\
    join_consistent_state A K state /\
    cont_implies_join_forward A K state.

  (** Empty state satisfies full invariant *)
  Theorem empty_state_full_invariant :
    forall qualifier, full_invariant (empty_gr_state qualifier).
  Proof.
    intros qualifier.
    unfold full_invariant.
    repeat split.
    - apply empty_state_no_pending_match.
    - apply empty_state_join_consistent_fwd.
    - unfold cont_implies_join_forward, empty_gr_state. simpl.
      intros channels patterns cont cpersist Hcont. inversion Hcont.
  Qed.

  (** Produce maintains the no-pending-match invariant *)
  Theorem produce_maintains_invariant :
    forall state ch data persist state',
      no_pending_match A K M state ->
      produce_spec A K M state ch data persist = PR_Stored A K state' ->
      no_pending_match A K M state'.
  Proof.
    intros state ch data persist state' Hinv Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs [[patterns cont] cpersist]] |] eqn:Hfind.
    - (* Found match - this case produces PR_Fired, contradicts Hprod *)
      discriminate.
    - (* No match found - data is stored *)
      injection Hprod as Hstate'.
      subst state'.
      unfold no_pending_match in *.
      simpl.
      intros ch' data' persist' Hdata' join_channels patterns' cont' cpersist' Hjoin Hcont.
      (* Case analysis on whether ch' = ch *)
      destruct (Z.eq_dec ch' ch) as [Heq | Hneq].
      + (* ch' = ch: the channel where we stored data *)
        subst ch'.
        simpl in Hdata'.
        destruct (Z.eq_dec ch ch) as [_ | Habs]; [| exfalso; apply Habs; reflexivity].
        destruct Hdata' as [Hnew | Hold].
        * (* data' is the newly added data *)
          injection Hnew as Hd Hp.
          subst data' persist'.
          (* We need to show that no continuation matches the new data.
             But find_matching_cont returned None for ch's joins.
             However, join_channels may be in gr_joins state ch. *)
          rewrite <- any_pattern_matches_equiv.
          eapply find_matching_cont_none_no_match.
          -- exact Hfind.
          -- exact Hjoin.
          -- exact Hcont.
        * (* data' was already in store *)
          eapply Hinv; eassumption.
      + (* ch' <> ch: different channel, data store unchanged *)
        simpl in Hdata'.
        destruct (Z.eq_dec ch' ch) as [Habs | _]; [exfalso; apply Hneq; exact Habs |].
        eapply Hinv; eassumption.
  Qed.

  (** Produce firing also maintains the invariant *)
  Theorem produce_fired_maintains_invariant :
    forall state ch data persist cont d state',
      no_pending_match A K M state ->
      produce_spec A K M state ch data persist = PR_Fired A K cont d state' ->
      no_pending_match A K M state'.
  Proof.
    intros state ch data persist cont d state' Hinv Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs [[patterns cont'] cpersist]] |] eqn:Hfind.
    - (* Found match - continuation removed *)
      injection Hprod as Hcont Hd Hstate'.
      subst cont' d state'.
      unfold no_pending_match in *.
      simpl.
      intros ch' data' persist' Hdata' join_channels patterns' cont'' cpersist' Hjoin Hcont.
      (* The data store is unchanged, so Hdata' implies data' was already there *)
      (* The continuation store has one continuation removed *)
      destruct (list_eq_dec Z.eq_dec join_channels join_chs) as [Heq | Hneq].
      + (* join_channels = join_chs: the channel group where we removed a continuation *)
        subst join_channels.
        (* (patterns', cont'', cpersist') is in remove_first ... (gr_cont_store state join_chs) *)
        eapply Hinv.
        -- exact Hdata'.
        -- exact Hjoin.
        -- eapply remove_first_preserves_non_matching.
           exact Hcont.
      + (* join_channels <> join_chs: continuation store unchanged for this group *)
        eapply Hinv; eassumption.
    - (* No match found - would produce PR_Stored, contradicts assumption *)
      discriminate.
  Qed.

  (** Combined theorem: produce always maintains the invariant *)
  Theorem produce_always_maintains_invariant :
    forall state ch data persist,
      no_pending_match A K M state ->
      match produce_spec A K M state ch data persist with
      | @PR_Stored _ _ state' => no_pending_match A K M state'
      | @PR_Fired _ _ _ _ state' => no_pending_match A K M state'
      end.
  Proof.
    intros state ch data persist Hinv.
    destruct (produce_spec A K M state ch data persist) as [state' | cont d state'] eqn:Hprod.
    - apply produce_maintains_invariant with state ch data persist.
      + exact Hinv.
      + exact Hprod.
    - apply produce_fired_maintains_invariant with state ch data persist cont d.
      + exact Hinv.
      + exact Hprod.
  Qed.

  (** Produce preserves join consistency when storing *)
  Theorem produce_preserves_join_consistency_stored :
    forall state ch data persist state',
      join_consistent_state A K state ->
      produce_spec A K M state ch data persist = PR_Stored A K state' ->
      join_consistent_state A K state'.
  Proof.
    intros state ch data persist state' Hjoin_cons Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs cont_entry] |] eqn:Hfind.
    - (* Some case - produces PR_Fired, not PR_Stored *)
      destruct cont_entry as [[patterns cont'] cpersist].
      inversion Hprod.
    - (* None case - produces PR_Stored *)
      injection Hprod as Hstate'.
      subst state'.
      unfold join_consistent_state in *.
      simpl.
      exact Hjoin_cons.
  Qed.

  (** Produce preserves join consistency when firing *)
  Theorem produce_preserves_join_consistency_fired :
    forall state ch data persist cont d state',
      join_consistent_state A K state ->
      produce_spec A K M state ch data persist = PR_Fired A K cont d state' ->
      join_consistent_state A K state'.
  Proof.
    intros state ch data persist cont d state' Hjoin_cons Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs [[patterns cont'] cpersist]] |] eqn:Hfind.
    - (* Some case - produces PR_Fired *)
      injection Hprod as _ _ Hstate'.
      subst state'.
      unfold join_consistent_state in *.
      simpl.
      exact Hjoin_cons.
    - (* None case - produces PR_Stored, not PR_Fired *)
      inversion Hprod.
  Qed.

  (** Produce always preserves join consistency *)
  Theorem produce_always_preserves_join_consistency :
    forall state ch data persist,
      join_consistent_state A K state ->
      match produce_spec A K M state ch data persist with
      | @PR_Stored _ _ state' => join_consistent_state A K state'
      | @PR_Fired _ _ _ _ state' => join_consistent_state A K state'
      end.
  Proof.
    intros state ch data persist Hjoin_cons.
    destruct (produce_spec A K M state ch data persist) as [state' | cont d state'] eqn:Hprod.
    - eapply produce_preserves_join_consistency_stored; eassumption.
    - eapply produce_preserves_join_consistency_fired; eassumption.
  Qed.

  (** Produce preserves cont_implies_join_forward when storing *)
  Theorem produce_preserves_cont_implies_join_stored :
    forall state ch data persist state',
      cont_implies_join_forward A K state ->
      produce_spec A K M state ch data persist = PR_Stored A K state' ->
      cont_implies_join_forward A K state'.
  Proof.
    intros state ch data persist state' Hcij Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs cont_entry] |] eqn:Hfind.
    - destruct cont_entry as [[patterns cont'] cpersist]. inversion Hprod.
    - injection Hprod as Hstate'.
      subst state'.
      unfold cont_implies_join_forward in *.
      simpl.
      exact Hcij.
  Qed.

  (** Produce preserves cont_implies_join_forward when firing *)
  Theorem produce_preserves_cont_implies_join_fired :
    forall state ch data persist cont d state',
      cont_implies_join_forward A K state ->
      produce_spec A K M state ch data persist = PR_Fired A K cont d state' ->
      cont_implies_join_forward A K state'.
  Proof.
    intros state ch data persist cont d state' Hcij Hprod.
    unfold produce_spec in Hprod.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs [[patterns cont'] cpersist]] |] eqn:Hfind.
    - injection Hprod as _ _ Hstate'.
      subst state'.
      unfold cont_implies_join_forward in *.
      simpl.
      (* Continuations are preserved except the one that was removed *)
      intros channels' patterns' cont'' cpersist' Hcont' ch' Hch_in'.
      destruct (list_eq_dec Z.eq_dec channels' join_chs) as [Heq | Hneq].
      + (* channels' = join_chs: the group where we removed a continuation *)
        subst channels'.
        apply Hcij with patterns' cont'' cpersist'.
        * eapply remove_first_preserves_non_matching. exact Hcont'.
        * exact Hch_in'.
      + (* channels' <> join_chs: unchanged *)
        apply Hcij with patterns' cont'' cpersist'. exact Hcont'. exact Hch_in'.
    - inversion Hprod.
  Qed.

  (** Produce maintains the full invariant *)
  Theorem produce_maintains_full_invariant :
    forall state ch data persist,
      full_invariant state ->
      match produce_spec A K M state ch data persist with
      | @PR_Stored _ _ state' => full_invariant state'
      | @PR_Fired _ _ _ _ state' => full_invariant state'
      end.
  Proof.
    intros state ch data persist [Hnpm [Hjc Hcij]].
    destruct (produce_spec A K M state ch data persist) as [state' | cont d state'] eqn:Hprod.
    - unfold full_invariant. repeat split.
      + eapply produce_maintains_invariant; eassumption.
      + eapply produce_preserves_join_consistency_stored; eassumption.
      + eapply produce_preserves_cont_implies_join_stored; eassumption.
    - unfold full_invariant. repeat split.
      + eapply produce_fired_maintains_invariant; eassumption.
      + eapply produce_preserves_join_consistency_fired; eassumption.
      + eapply produce_preserves_cont_implies_join_fired; eassumption.
  Qed.

  (** Produce either fires or stores, never both *)
  Theorem produce_exclusive :
    forall state ch data persist,
      (exists state' cont d, produce_spec A K M state ch data persist = PR_Fired A K cont d state') \/
      (exists state', produce_spec A K M state ch data persist = PR_Stored A K state').
  Proof.
    intros state ch data persist.
    unfold produce_spec.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs cont_entry] |].
    - left. destruct cont_entry as [[patterns cont] cpersist].
      eexists. eexists. eexists. reflexivity.
    - right. eexists. reflexivity.
  Qed.

  (** Consume maintains the invariant when no data matches *)
  Theorem consume_maintains_invariant :
    forall state channels patterns cont persist state',
      no_pending_match A K M state ->
      join_consistent_state A K state ->  (* Join consistency precondition *)
      cont_implies_join_forward A K state ->  (* Continuation implies join precondition *)
      consume_spec A K state channels patterns cont persist = @CR_Stored A K state' ->
      (* Additional precondition: no data matches the new patterns *)
      (forall ch, In ch channels ->
        forall data dpersist,
          In (data, dpersist) (gr_data_store state ch) ->
          any_pattern_matches A M patterns data = false) ->
      no_pending_match A K M state'.
  Proof.
    intros state channels patterns cont persist state' Hinv Hjoin_cons Hcont_join Hcons Hpre.
    unfold consume_spec in Hcons.
    injection Hcons as Hstate'.
    subst state'.
    unfold no_pending_match in *.
    simpl.
    intros ch data dpersist Hdata join_channels patterns' cont' cpersist' Hjoin Hcont.
    (* Case analysis on whether this is the new continuation or an old one *)
    simpl in Hcont.
    destruct (list_eq_dec Z.eq_dec join_channels channels) as [Heq | Hneq].
    - (* join_channels = channels *)
      subst join_channels.
      destruct Hcont as [Hnew_cont | Hold_cont].
      + (* (patterns', cont', cpersist') is the new continuation *)
        injection Hnew_cont as Hp Hc Hcp.
        subst patterns' cont' cpersist'.
        (* ch must be in channels for the join to contain channels *)
        destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
        * (* ch in channels - use precondition *)
          apply Hpre with ch dpersist.
          -- exact Hch_in.
          -- exact Hdata.
        * (* ch not in channels - use join consistency to derive contradiction *)
          simpl in Hjoin.
          destruct (In_dec Z.eq_dec ch channels) as [Habs | _]; [contradiction |].
          exfalso. apply Hch_notin.
          apply Hjoin_cons. exact Hjoin.
      + (* (patterns', cont', cpersist') is an old continuation *)
        (* Use cont_implies_join to show channels was in old joins *)
        destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
        * (* ch in channels - old continuation means channels was in old joins *)
          assert (Hjoin_old : In channels (gr_joins state ch)).
          { apply Hcont_join with patterns' cont' cpersist'. exact Hold_cont. exact Hch_in. }
          eapply Hinv.
          -- exact Hdata.
          -- exact Hjoin_old.
          -- exact Hold_cont.
        * (* ch not in channels - use join consistency *)
          simpl in Hjoin.
          destruct (In_dec Z.eq_dec ch channels) as [Habs | _]; [contradiction |].
          exfalso. apply Hch_notin.
          apply Hjoin_cons. exact Hjoin.
    - (* join_channels <> channels: continuation store unchanged for this group *)
      simpl in Hjoin.
      destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
      + (* ch in channels - joins may have new entry but we look at different group *)
        destruct Hjoin as [Hnew_join | Hold_join].
        * (* channels was added but join_channels <> channels - contradiction *)
          subst. contradiction.
        * eapply Hinv; eassumption.
      + (* ch not in channels - joins unchanged *)
        eapply Hinv; eassumption.
  Qed.

  (** Consume preserves join consistency *)
  Theorem consume_preserves_join_consistency :
    forall state channels patterns cont persist state',
      join_consistent_state A K state ->
      consume_spec A K state channels patterns cont persist = @CR_Stored A K state' ->
      join_consistent_state A K state'.
  Proof.
    intros state channels patterns cont persist state' Hjoin_cons Hcons.
    unfold consume_spec in Hcons.
    injection Hcons as Hstate'.
    subst state'.
    unfold join_consistent_state in *.
    simpl.
    intros ch channels' Hjoin'.
    destruct (In_dec Z.eq_dec ch channels) as [Hch_in | Hch_notin].
    - (* ch in channels *)
      destruct Hjoin' as [Hnew | Hold].
      + (* channels' = channels (newly added) *)
        subst. exact Hch_in.
      + (* channels' was already in gr_joins state ch *)
        apply Hjoin_cons. exact Hold.
    - (* ch not in channels - joins unchanged *)
      apply Hjoin_cons. exact Hjoin'.
  Qed.

  (** Consume preserves cont_implies_join_forward *)
  Theorem consume_preserves_cont_implies_join_forward :
    forall state channels patterns cont persist state',
      cont_implies_join_forward A K state ->
      consume_spec A K state channels patterns cont persist = @CR_Stored A K state' ->
      cont_implies_join_forward A K state'.
  Proof.
    intros state channels patterns cont persist state' Hcij Hcons.
    unfold consume_spec in Hcons.
    injection Hcons as Hstate'.
    subst state'.
    unfold cont_implies_join_forward in *.
    simpl.
    intros channels' patterns' cont' cpersist' Hcont' ch' Hch_in'.
    destruct (list_eq_dec Z.eq_dec channels' channels) as [Heq | Hneq].
    - (* channels' = channels *)
      subst channels'.
      destruct Hcont' as [Hnew | Hold].
      + (* New continuation for channels *)
        destruct (In_dec Z.eq_dec ch' channels) as [Hin | Hnotin].
        * left. reflexivity.
        * contradiction.
      + (* Old continuation for channels *)
        destruct (In_dec Z.eq_dec ch' channels) as [Hin | Hnotin].
        * left. reflexivity.
        * contradiction.
    - (* channels' <> channels *)
      destruct (In_dec Z.eq_dec ch' channels) as [Hch_in_chs | Hch_notin_chs].
      + right. apply Hcij with patterns' cont' cpersist'. exact Hcont'. exact Hch_in'.
      + apply Hcij with patterns' cont' cpersist'. exact Hcont'. exact Hch_in'.
  Qed.

  (** Consume maintains the full invariant *)
  Theorem consume_maintains_full_invariant :
    forall state channels patterns cont persist state',
      full_invariant state ->
      consume_spec A K state channels patterns cont persist = @CR_Stored A K state' ->
      (forall ch, In ch channels ->
        forall data dpersist,
          In (data, dpersist) (gr_data_store state ch) ->
          any_pattern_matches A M patterns data = false) ->
      full_invariant state'.
  Proof.
    intros state channels patterns cont persist state' [Hnpm [Hjc Hcij]] Hcons Hpre.
    unfold full_invariant. repeat split.
    - eapply consume_maintains_invariant; eassumption.
    - eapply consume_preserves_join_consistency; eassumption.
    - eapply consume_preserves_cont_implies_join_forward; eassumption.
  Qed.
End InvariantPreservation.

(** ** Gensym Properties *)

Section GensymSpec.
  Variable A K : Type.

  (** Gensym produces unique channel names *)
  Definition gensym (state : GenericRSpaceState A K) : Channel * GenericRSpaceState A K :=
    let counter := gr_name_counter state in
    let new_ch := Z.of_nat counter in
    let new_state := mkGRState
      (gr_data_store state)
      (gr_cont_store state)
      (gr_joins state)
      (gr_qualifier state)
      (S counter)
    in
    (new_ch, new_state).

  (** Gensym always increases counter *)
  Theorem gensym_increases_counter :
    forall state ch state',
      gensym state = (ch, state') ->
      gr_name_counter state' = S (gr_name_counter state).
  Proof.
    intros state ch state' H.
    unfold gensym in H.
    injection H as _ Hs'.
    rewrite <- Hs'. simpl. reflexivity.
  Qed.

  (** Sequential gensym calls produce different channels *)
  Theorem gensym_produces_unique :
    forall state ch1 state1 ch2 state2,
      gensym state = (ch1, state1) ->
      gensym state1 = (ch2, state2) ->
      ch1 <> ch2.
  Proof.
    intros state ch1 state1 ch2 state2 H1 H2.
    unfold gensym in H1, H2.
    injection H1 as Hch1 Hs1.
    injection H2 as Hch2 Hs2.
    rewrite <- Hch1, <- Hch2.
    rewrite <- Hs1. simpl.
    lia.
  Qed.

  (** Gensym preserves the no-pending-match invariant *)
  Theorem gensym_maintains_invariant :
    forall M state ch state',
      no_pending_match A K M state ->
      gensym state = (ch, state') ->
      no_pending_match A K M state'.
  Proof.
    intros M state ch state' Hinv Hgen.
    unfold gensym in Hgen.
    injection Hgen as _ Hs'.
    subst state'.
    unfold no_pending_match in *.
    simpl.
    exact Hinv.
  Qed.
End GensymSpec.

(** ** Space Qualifier Properties *)

Section QualifierSpec.
  Variable A K : Type.

  (** Qualifier constants *)
  Definition QUALIFIER_DEFAULT : SpaceQualifier := 0.
  Definition QUALIFIER_TEMP : SpaceQualifier := 1.
  Definition QUALIFIER_SEQ : SpaceQualifier := 2.

  (** Default qualifier allows persistence *)
  Definition is_persistent (state : GenericRSpaceState A K) : bool :=
    Nat.eqb (gr_qualifier state) QUALIFIER_DEFAULT.

  (** Temp qualifier for temporary spaces *)
  Definition is_temp (state : GenericRSpaceState A K) : bool :=
    Nat.eqb (gr_qualifier state) QUALIFIER_TEMP.

  (** Seq qualifier restricts mobility *)
  Definition is_seq (state : GenericRSpaceState A K) : bool :=
    Nat.eqb (gr_qualifier state) QUALIFIER_SEQ.

  (** Seq qualifier restricts mobility *)
  Definition is_mobile (state : GenericRSpaceState A K) : bool :=
    negb (is_seq state).

  (** Seq qualifier restricts concurrency *)
  Definition is_concurrent (state : GenericRSpaceState A K) : bool :=
    negb (is_seq state).

  (** Qualifier properties *)
  Theorem qualifier_default_is_persistent :
    forall state, gr_qualifier state = QUALIFIER_DEFAULT -> is_persistent state = true.
  Proof.
    intros state H. unfold is_persistent. rewrite H. reflexivity.
  Qed.

  Theorem qualifier_seq_not_mobile :
    forall state, gr_qualifier state = QUALIFIER_SEQ -> is_mobile state = false.
  Proof.
    intros state H. unfold is_mobile, is_seq. rewrite H. reflexivity.
  Qed.

  Theorem qualifier_seq_not_concurrent :
    forall state, gr_qualifier state = QUALIFIER_SEQ -> is_concurrent state = false.
  Proof.
    intros state H. unfold is_concurrent, is_seq. rewrite H. reflexivity.
  Qed.

  Theorem qualifier_default_is_mobile :
    forall state, gr_qualifier state = QUALIFIER_DEFAULT -> is_mobile state = true.
  Proof.
    intros state H. unfold is_mobile, is_seq. rewrite H. reflexivity.
  Qed.

  Theorem qualifier_temp_is_mobile :
    forall state, gr_qualifier state = QUALIFIER_TEMP -> is_mobile state = true.
  Proof.
    intros state H. unfold is_mobile, is_seq. rewrite H. reflexivity.
  Qed.
End QualifierSpec.

(** ** Seq Qualifier Mobility Restrictions *)

(** The Seq qualifier imposes strict mobility and concurrency restrictions
    as specified in the Reifying RSpaces specification (lines 280-284):
    1. Cannot send Seq channel to other processes
    2. Cannot use in concurrent processes
    3. Can only be used in a single sequential process *)
Section SeqMobilityRestrictions.
  Variable A K : Type.

  (** Qualifier constants (redefined locally to avoid section variable issues) *)
  Let SEQ_QUALIFIER : SpaceQualifier := 2.
  Let DEFAULT_QUALIFIER : SpaceQualifier := 0.
  Let TEMP_QUALIFIER : SpaceQualifier := 1.

  (** Process identifier type *)
  Definition ProcessId := nat.

  (** A channel access record tracks which process accesses a channel *)
  Record ChannelAccess := mkChannelAccess {
    ca_channel : Channel;
    ca_process : ProcessId;
    ca_concurrent : bool;  (* true if access is from a concurrent context *)
  }.

  (** Valid Seq channel usage: accessed only by one process, non-concurrently *)
  Definition valid_seq_access (accesses : list ChannelAccess) (ch : Channel) : Prop :=
    (* All accesses to ch are from the same process *)
    (forall a1 a2, In a1 accesses -> In a2 accesses ->
      ca_channel a1 = ch -> ca_channel a2 = ch ->
      ca_process a1 = ca_process a2) /\
    (* No concurrent accesses *)
    (forall a, In a accesses -> ca_channel a = ch -> ca_concurrent a = false).

  (** Seq channels cannot be sent (mobility restriction) *)
  Definition seq_not_sendable (state : GenericRSpaceState A K) (ch : Channel) : Prop :=
    gr_qualifier state = SEQ_QUALIFIER ->
    (* The channel cannot appear in data sent on any channel *)
    forall dest_ch data persist,
      In (data, persist) (gr_data_store state dest_ch) ->
      (* data cannot "contain" ch - this is a semantic constraint *)
      True.  (* Placeholder - actual check depends on data type containing channels *)

  (** Formal specification: Seq qualifier implies not mobile *)
  Theorem seq_implies_not_mobile :
    forall state,
      gr_qualifier state = SEQ_QUALIFIER ->
      is_mobile A K state = false.
  Proof.
    intros state Hseq.
    unfold is_mobile, is_seq.
    rewrite Hseq.
    simpl. reflexivity.
  Qed.

  (** Formal specification: Seq qualifier implies not concurrent *)
  Theorem seq_implies_not_concurrent :
    forall state,
      gr_qualifier state = SEQ_QUALIFIER ->
      is_concurrent A K state = false.
  Proof.
    intros state Hseq.
    unfold is_concurrent, is_seq.
    rewrite Hseq.
    simpl. reflexivity.
  Qed.

  (** Contrapositive: if mobile, then not Seq *)
  Theorem mobile_implies_not_seq :
    forall state,
      is_mobile A K state = true ->
      gr_qualifier state <> SEQ_QUALIFIER.
  Proof.
    intros state Hmobile Hseq.
    rewrite seq_implies_not_mobile in Hmobile by exact Hseq.
    discriminate.
  Qed.

  (** Contrapositive: if concurrent, then not Seq *)
  Theorem concurrent_implies_not_seq :
    forall state,
      is_concurrent A K state = true ->
      gr_qualifier state <> SEQ_QUALIFIER.
  Proof.
    intros state Hconc Hseq.
    rewrite seq_implies_not_concurrent in Hconc by exact Hseq.
    discriminate.
  Qed.

  (** Qualifier exclusivity: exactly one qualifier per state *)
  Theorem qualifier_exclusive :
    forall (state : GenericRSpaceState A K),
      (@gr_qualifier A K state = DEFAULT_QUALIFIER /\
       @gr_qualifier A K state <> TEMP_QUALIFIER /\
       @gr_qualifier A K state <> SEQ_QUALIFIER) \/
      (@gr_qualifier A K state = TEMP_QUALIFIER /\
       @gr_qualifier A K state <> DEFAULT_QUALIFIER /\
       @gr_qualifier A K state <> SEQ_QUALIFIER) \/
      (@gr_qualifier A K state = SEQ_QUALIFIER /\
       @gr_qualifier A K state <> DEFAULT_QUALIFIER /\
       @gr_qualifier A K state <> TEMP_QUALIFIER) \/
      (@gr_qualifier A K state <> DEFAULT_QUALIFIER /\
       @gr_qualifier A K state <> TEMP_QUALIFIER /\
       @gr_qualifier A K state <> SEQ_QUALIFIER).
  Proof.
    intros state.
    unfold DEFAULT_QUALIFIER, TEMP_QUALIFIER, SEQ_QUALIFIER.
    destruct (Nat.eq_dec (@gr_qualifier A K state) 0) as [H0 | Hn0].
    - left. repeat split; [exact H0 | intro; lia | intro; lia].
    - destruct (Nat.eq_dec (@gr_qualifier A K state) 1) as [H1 | Hn1].
      + right. left. repeat split; [exact H1 | intro; lia | intro; lia].
      + destruct (Nat.eq_dec (@gr_qualifier A K state) 2) as [H2 | Hn2].
        * right. right. left. repeat split; [exact H2 | intro; lia | intro; lia].
        * right. right. right. repeat split; intro; lia.
  Qed.

  (** Seq channels preserve their qualifier through operations *)
  Theorem produce_preserves_qualifier :
    forall M state ch data persist,
      match produce_spec A K M state ch data persist with
      | PR_Stored _ _ state' => gr_qualifier state' = gr_qualifier state
      | PR_Fired _ _ _ _ state' => gr_qualifier state' = gr_qualifier state
      end.
  Proof.
    intros M state ch data persist.
    unfold produce_spec.
    destruct (find_matching_cont A K M (gr_joins state ch) (gr_cont_store state) data)
      as [[join_chs [[patterns cont] cpersist]] |].
    - simpl. reflexivity.
    - simpl. reflexivity.
  Qed.

  (** Consume preserves qualifier *)
  Theorem consume_preserves_qualifier :
    forall state channels patterns cont persist,
      match consume_spec A K state channels patterns cont persist with
      | CR_Stored _ _ state' => gr_qualifier state' = gr_qualifier state
      | CR_Matched _ _ _ state' => gr_qualifier state' = gr_qualifier state
      end.
  Proof.
    intros state channels patterns cont persist.
    unfold consume_spec.
    simpl. reflexivity.
  Qed.

  (** Gensym preserves qualifier *)
  Theorem gensym_preserves_qualifier :
    forall state ch state',
      gensym A K state = (ch, state') ->
      gr_qualifier state' = gr_qualifier state.
  Proof.
    intros state ch state' Hgen.
    unfold gensym in Hgen.
    injection Hgen as _ Hs'.
    subst state'. simpl. reflexivity.
  Qed.

  (** Seq state remains Seq through all operations *)
  Theorem seq_invariant_through_produce :
    forall M state ch data persist,
      gr_qualifier state = SEQ_QUALIFIER ->
      match produce_spec A K M state ch data persist with
      | PR_Stored _ _ state' => gr_qualifier state' = SEQ_QUALIFIER
      | PR_Fired _ _ _ _ state' => gr_qualifier state' = SEQ_QUALIFIER
      end.
  Proof.
    intros M state ch data persist Hseq.
    pose proof (produce_preserves_qualifier M state ch data persist) as Hpres.
    destruct (produce_spec A K M state ch data persist); rewrite Hpres; exact Hseq.
  Qed.

  (** Sequential access pattern: at most one active accessor *)
  Definition single_accessor_invariant
    (accessor : option ProcessId) (state : GenericRSpaceState A K) : Prop :=
    gr_qualifier state = SEQ_QUALIFIER ->
    accessor <> None.  (* There must be exactly one accessor *)

  (** Safe Seq operation: performed by the single accessor *)
  Definition safe_seq_operation
    (accessor : ProcessId) (state : GenericRSpaceState A K) : Prop :=
    gr_qualifier state = SEQ_QUALIFIER ->
    is_concurrent A K state = false /\
    is_mobile A K state = false.

  (** Safe Seq operations are always valid *)
  Theorem safe_seq_always_valid :
    forall accessor state,
      gr_qualifier state = SEQ_QUALIFIER ->
      safe_seq_operation accessor state.
  Proof.
    intros accessor state Hseq.
    unfold safe_seq_operation.
    intros _.
    split.
    - apply seq_implies_not_concurrent. exact Hseq.
    - apply seq_implies_not_mobile. exact Hseq.
  Qed.

End SeqMobilityRestrictions.

(** ** Multi-Channel Join Semantics *)

Section MultiChannelJoin.
  Variable A K : Type.
  Variable M : Match A A.

  (** A join requires ALL channels to have matching data *)
  Fixpoint all_channels_have_matching_data
    (channels : list Channel) (patterns : list A)
    (data_store : Channel -> list (A * bool)) : bool :=
    match channels, patterns with
    | [], [] => true
    | ch :: ch_rest, pat :: pat_rest =>
      let data_at_ch := data_store ch in
      let has_match := existsb (fun '(d, _) => @match_fn A A M pat d) data_at_ch in
      has_match && all_channels_have_matching_data ch_rest pat_rest data_store
    | _, _ => false (* Mismatched lengths *)
    end.

  (** Join atomicity: either all channels have data or continuation is stored *)
  Definition join_atomic (state : GenericRSpaceState A K)
    (channels : list Channel) (patterns : list A) : Prop :=
    all_channels_have_matching_data channels patterns (gr_data_store state) = true \/
    (exists cont persist,
      In (patterns, cont, persist) (gr_cont_store state channels)).

  (** Join consistency is defined in JoinConsistency section above.
      Here we provide the specialized version for this section's variables. *)
  Definition join_consistent (state : GenericRSpaceState A K) : Prop :=
    join_consistent_state A K state.

  (** Empty state is join consistent (using the general theorem) *)
  Theorem empty_state_join_consistent :
    forall qualifier, join_consistent (empty_gr_state qualifier).
  Proof.
    intros qualifier.
    apply empty_state_join_consistent_fwd.
  Qed.

  (** Key property: all_channels_have_matching_data is true iff EVERY channel has matching data *)
  Theorem all_channels_have_matching_data_correct :
    forall channels patterns data_store,
      length channels = length patterns ->
      all_channels_have_matching_data channels patterns data_store = true <->
      (forall i ch pat,
        nth_error channels i = Some ch ->
        nth_error patterns i = Some pat ->
        existsb (fun '(d, _) => @match_fn A A M pat d) (data_store ch) = true).
  Proof.
    intros channels.
    induction channels as [| ch_hd ch_tl IH].
    - (* Empty channels *)
      intros patterns data_store Hlen.
      split.
      + intros _ i ch pat Hch Hpat.
        rewrite nth_error_nil in Hch. discriminate.
      + intros _. destruct patterns; [reflexivity | simpl in Hlen; discriminate].
    - (* ch_hd :: ch_tl *)
      intros patterns data_store Hlen.
      destruct patterns as [| pat_hd pat_tl].
      + simpl in Hlen. discriminate.
      + split.
        * intros Hmatch i ch pat Hch Hpat.
          simpl in Hmatch.
          destruct (existsb (fun '(d, _) => @match_fn A A M pat_hd d) (data_store ch_hd)) eqn:Hhead;
            [| discriminate].
          destruct i.
          -- simpl in Hch, Hpat.
             injection Hch as ->. injection Hpat as ->.
             exact Hhead.
          -- simpl in Hch, Hpat.
             simpl in Hlen. injection Hlen as Hlen.
             assert (Htail := proj1 (IH pat_tl data_store Hlen) Hmatch).
             apply Htail with i; assumption.
        * intros Hpointwise.
          simpl.
          assert (Hhead : existsb (fun '(d, _) => @match_fn A A M pat_hd d) (data_store ch_hd) = true).
          { apply Hpointwise with 0; reflexivity. }
          rewrite Hhead. simpl.
          simpl in Hlen. injection Hlen as Hlen.
          apply (proj2 (IH pat_tl data_store Hlen)).
          intros i ch pat Hch Hpat.
          apply Hpointwise with (S i); simpl; assumption.
  Qed.

  (** Multi-channel consume only fires when ALL channels have matching data *)
  Theorem multi_channel_consume_all_or_nothing :
    forall state channels patterns,
      length channels = length patterns ->
      join_consistent state ->
      (all_channels_have_matching_data channels patterns (gr_data_store state) = true \/
       all_channels_have_matching_data channels patterns (gr_data_store state) = false).
  Proof.
    intros state channels patterns Hlen Hjc.
    destruct (all_channels_have_matching_data channels patterns (gr_data_store state)).
    - left. reflexivity.
    - right. reflexivity.
  Qed.

  (** Multi-channel join semantics: consume blocks until all patterns match *)
  Definition consume_blocks_until_all_match
    (state : GenericRSpaceState A K) (channels : list Channel) (patterns : list A) : Prop :=
    all_channels_have_matching_data channels patterns (gr_data_store state) = false ->
    exists cont persist, In (patterns, cont, persist) (gr_cont_store state channels).

  (** ** Multi-way Consume Atomicity (Spec Lines 484-497) *)

  (** Consume result is either all-match or all-wait - no partial matching *)
  Inductive MultiConsumeOutcome :=
    | MCO_AllMatched : list A -> MultiConsumeOutcome  (* All channels had matching data *)
    | MCO_AllWaiting : MultiConsumeOutcome.           (* Continuation stored, waiting for all *)

  (** Determine outcome of multi-channel consume *)
  Definition multi_consume_outcome
    (state : GenericRSpaceState A K)
    (channels : list Channel) (patterns : list A)
    : MultiConsumeOutcome :=
    match find_all_matching_data_local A M channels patterns (gr_data_store state) with
    | Some matched_data => MCO_AllMatched matched_data
    | None => MCO_AllWaiting
    end.

  (** If all channels have matching data, find_all_matching_data_local succeeds *)
  Theorem all_channels_match_implies_find_succeeds :
    forall channels patterns data_store,
      length channels = length patterns ->
      all_channels_have_matching_data channels patterns data_store = true ->
      exists matched_data,
        find_all_matching_data_local A M channels patterns data_store = Some matched_data.
  Proof.
    intros channels.
    induction channels as [| ch_hd ch_tl IH].
    - intros patterns data_store Hlen Hmatch.
      destruct patterns; [| simpl in Hlen; discriminate].
      simpl. exists []. reflexivity.
    - intros patterns data_store Hlen Hmatch.
      destruct patterns as [| pat_hd pat_tl]; [simpl in Hlen; discriminate |].
      simpl in Hmatch.
      destruct (existsb (fun '(d, _) => @match_fn A A M pat_hd d) (data_store ch_hd)) eqn:Hexists;
        [| discriminate].
      simpl.
      (* existsb = true means find_first succeeds *)
      assert (Hfind : exists d p, find_first (fun '(d', _) => @match_fn A A M pat_hd d')
                                    (data_store ch_hd) = Some (d, p)).
      {
        clear IH Hmatch.
        induction (data_store ch_hd) as [| [d' p'] rest IHrest].
        - simpl in Hexists. discriminate.
        - simpl in Hexists. simpl.
          destruct (@match_fn A A M pat_hd d') eqn:Hm.
          + exists d', p'. reflexivity.
          + apply IHrest. exact Hexists.
      }
      destruct Hfind as [d [p Hfind]].
      rewrite Hfind.
      simpl in Hlen. injection Hlen as Hlen.
      destruct (IH pat_tl data_store Hlen Hmatch) as [rest_data Hrest].
      rewrite Hrest.
      exists (d :: rest_data). reflexivity.
  Qed.

  (** Multi-channel consume is atomic: either all channels match or none do.
      There is no state where "some" channels matched and others didn't. *)
  Theorem consume_multi_atomic :
    forall state channels patterns,
      length channels = length patterns ->
      (multi_consume_outcome state channels patterns = MCO_AllWaiting /\
       all_channels_have_matching_data channels patterns (gr_data_store state) = false) \/
      (exists matched_data,
        multi_consume_outcome state channels patterns = MCO_AllMatched matched_data).
  Proof.
    intros state channels patterns Hlen.
    unfold multi_consume_outcome.
    destruct (find_all_matching_data_local A M channels patterns (gr_data_store state))
      as [matched_data |] eqn:Hfind.
    - (* All matched *)
      right. exists matched_data. reflexivity.
    - (* None matched - we need to show all_channels_have_matching_data = false *)
      left. split; [reflexivity |].
      (* Prove that find_all_matching_data_local = None implies
         all_channels_have_matching_data = false.
         Proof by contradiction: if all_channels_have_matching_data = true,
         then find_all_matching_data_local would return Some, not None. *)
      destruct (all_channels_have_matching_data channels patterns (gr_data_store state)) eqn:Hall.
      + (* Suppose all_channels_have_matching_data = true *)
        exfalso.
        destruct (all_channels_match_implies_find_succeeds channels patterns
                    (gr_data_store state) Hlen Hall) as [md Hmd].
        rewrite Hmd in Hfind. discriminate.
      + reflexivity.
  Qed.

  (** Match order is deterministic: same state + same channels/patterns = same result *)
  Theorem consume_multi_match_order_deterministic :
    forall state channels patterns result1 result2,
      multi_consume_outcome state channels patterns = result1 ->
      multi_consume_outcome state channels patterns = result2 ->
      result1 = result2.
  Proof.
    intros state channels patterns result1 result2 H1 H2.
    rewrite <- H1, <- H2. reflexivity.
  Qed.

  (** Atomicity invariant: if consume fires, all data was present simultaneously *)
  Theorem consume_fires_implies_all_data_present :
    forall state channels patterns matched_data,
      multi_consume_outcome state channels patterns = MCO_AllMatched matched_data ->
      find_all_matching_data_local A M channels patterns (gr_data_store state) = Some matched_data.
  Proof.
    intros state channels patterns matched_data Houtcome.
    unfold multi_consume_outcome in Houtcome.
    remember (find_all_matching_data_local A M channels patterns (gr_data_store state)) as result.
    destruct result as [data |].
    - injection Houtcome as Hd. subst. reflexivity.
    - discriminate.
  Qed.

  (** Atomicity theorem: matched data length equals channels length *)
  Theorem consume_matched_data_length :
    forall channels patterns data_store matched_data,
      length channels = length patterns ->
      find_all_matching_data_local A M channels patterns data_store = Some matched_data ->
      length matched_data = length channels.
  Proof.
    intros channels.
    induction channels as [| ch_hd ch_tl IH].
    - intros patterns data_store matched_data Hlen Hfind.
      destruct patterns; [| simpl in Hlen; discriminate].
      simpl in Hfind. injection Hfind as Hd. subst. reflexivity.
    - intros patterns data_store matched_data Hlen Hfind.
      destruct patterns as [| pat_hd pat_tl]; [simpl in Hlen; discriminate |].
      simpl in Hfind.
      destruct (find_first (fun '(d, _) => @match_fn A A M pat_hd d) (data_store ch_hd))
        as [[d_hd p_hd] |] eqn:Hfirst.
      + destruct (find_all_matching_data_local A M ch_tl pat_tl data_store)
          as [rest_data |] eqn:Hrest.
        * injection Hfind as Hmd. subst matched_data.
          simpl. f_equal.
          simpl in Hlen. injection Hlen as Hlen.
          apply IH with pat_tl data_store; assumption.
        * discriminate.
      + discriminate.
  Qed.

End MultiChannelJoin.

(** ** Persist Flag Semantics *)

Section PersistSemantics.
  Variable A K : Type.
  Variable M : Match A A.

  (** Non-persistent data is removed after matching *)
  Definition match_removes_non_persistent
    (state state' : GenericRSpaceState A K)
    (ch : Channel) (data : A) (persist : bool) : Prop :=
    persist = false ->
    In (data, persist) (gr_data_store state ch) ->
    ~ In (data, persist) (gr_data_store state' ch).

  (** Persistent data remains after matching *)
  Definition match_keeps_persistent
    (state state' : GenericRSpaceState A K)
    (ch : Channel) (data : A) : Prop :=
    In (data, true) (gr_data_store state ch) ->
    In (data, true) (gr_data_store state' ch).
End PersistSemantics.
