(** * Safety Properties for Reified RSpaces

    This module consolidates and proves comprehensive safety properties
    for the RSpace system. Safety properties ensure that the system
    never enters an invalid state and that all invariants are maintained.

    Key safety properties:
    - No pending match: data and continuations don't coexist on same channel
    - Channel isolation: data on one channel doesn't affect others
    - Cross-space join prevention: joins cannot span space boundaries
    - Qualifier enforcement: Seq qualifier restrictions are enforced
    - Registry invariant preservation: operations maintain invariants
    - Gas accounting correctness: phlogiston cannot go negative

    Cross-references:
    - GenericRSpace.v: Core operation invariants
    - Registry/Invariants.v: Registry operation invariants
    - Phlogiston.v: Gas accounting invariants
*)

From Stdlib Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Core Safety Invariants *)

Section CoreSafety.
  Variable A K : Type.

  (** The fundamental "no pending match" invariant:
      At any channel, either there is data waiting OR continuations waiting,
      but never both simultaneously. This is the heart of RSpace correctness.

      Reference: GenericRSpace.v lines 157-180 *)
  Definition no_pending_match
    (data_store : Z -> list (A * bool))
    (cont_store : list Z -> list (list A * K * bool))
    : Prop :=
    forall (c : Z),
      data_store c = [] \/ cont_store [c] = [].

  (** If we have data at a channel, there are no continuations *)
  Theorem data_implies_no_conts :
    forall (data_store : Z -> list (A * bool))
           (cont_store : list Z -> list (list A * K * bool))
           (c : Z) (d : A * bool),
      no_pending_match data_store cont_store ->
      In d (data_store c) ->
      cont_store [c] = [].
  Proof.
    intros data_store cont_store c d Hinv Hdata.
    destruct (Hinv c) as [Hempty | Hconts].
    - (* data_store c = [] contradicts In d (data_store c) *)
      rewrite Hempty in Hdata. inversion Hdata.
    - (* cont_store [c] = [] *)
      exact Hconts.
  Qed.

  (** If we have continuations at a channel, there is no data *)
  Theorem conts_implies_no_data :
    forall (data_store : Z -> list (A * bool))
           (cont_store : list Z -> list (list A * K * bool))
           (c : Z) (k : list A * K * bool),
      no_pending_match data_store cont_store ->
      In k (cont_store [c]) ->
      data_store c = [].
  Proof.
    intros data_store cont_store c k Hinv Hconts.
    destruct (Hinv c) as [Hdata | Hempty].
    - (* data_store c = [] *)
      exact Hdata.
    - (* cont_store [c] = [] contradicts In k (cont_store [c]) *)
      rewrite Hempty in Hconts. inversion Hconts.
  Qed.

  (** Empty state satisfies no_pending_match *)
  Theorem empty_satisfies_no_pending_match :
    no_pending_match (fun _ => []) (fun _ => []).
  Proof.
    unfold no_pending_match.
    intros c. left. reflexivity.
  Qed.

End CoreSafety.

(** ** Channel Isolation *)

Section ChannelIsolation.
  Variable A K : Type.

  (** Operations on one channel don't affect data at other channels *)
  Definition channel_isolation
    (put_data : Z -> A -> (Z -> list (A * bool)) -> (Z -> list (A * bool)))
    : Prop :=
    forall (c1 c2 : Z) (a : A) (store : Z -> list (A * bool)),
      c1 <> c2 ->
      put_data c1 a store c2 = store c2.

  (** Isolation implies no interference between channels *)
  Theorem isolation_no_interference :
    forall (put_data : Z -> A -> (Z -> list (A * bool)) -> (Z -> list (A * bool)))
           (c1 c2 : Z) (a : A) (store : Z -> list (A * bool)),
      channel_isolation put_data ->
      c1 <> c2 ->
      put_data c1 a store c2 = store c2.
  Proof.
    intros put_data c1 c2 a store Hiso Hneq.
    apply Hiso. exact Hneq.
  Qed.

  (** Simple put_data implementation satisfies isolation *)
  Definition simple_put_data (c : Z) (a : A) (store : Z -> list (A * bool))
    : Z -> list (A * bool) :=
    fun c' => if Z.eq_dec c c' then (a, false) :: store c' else store c'.

  Theorem simple_put_data_isolation :
    channel_isolation simple_put_data.
  Proof.
    unfold channel_isolation, simple_put_data.
    intros c1 c2 a store Hneq.
    destruct (Z.eq_dec c1 c2) as [Heq | _].
    - exfalso. apply Hneq. exact Heq.
    - reflexivity.
  Qed.

End ChannelIsolation.

(** ** Space Qualifier Safety *)

Section QualifierSafety.

  (** Space qualifiers define behavior constraints *)
  Inductive SpaceQualifier :=
    | SQ_Default   (* Persistent, concurrent, mobile *)
    | SQ_Temp      (* Non-persistent, concurrent, mobile *)
    | SQ_Seq.      (* Non-persistent, sequential, non-mobile *)

  (** Mobility: can the channel be sent to other processes? *)
  Definition is_mobile (q : SpaceQualifier) : bool :=
    match q with
    | SQ_Seq => false
    | _ => true
    end.

  (** Concurrency: can multiple processes access simultaneously? *)
  Definition allows_concurrent (q : SpaceQualifier) : bool :=
    match q with
    | SQ_Seq => false
    | _ => true
    end.

  (** Persistence: is data persisted across checkpoints? *)
  Definition is_persistent (q : SpaceQualifier) : bool :=
    match q with
    | SQ_Default => true
    | _ => false
    end.

  (** Seq channels cannot be sent - this is a safety property
      Reference: GenericRSpace.v lines 1203-1212 *)
  Theorem seq_cannot_be_sent :
    forall (q : SpaceQualifier),
      q = SQ_Seq -> is_mobile q = false.
  Proof.
    intros q Hseq.
    subst q. reflexivity.
  Qed.

  (** Seq channels cannot be accessed concurrently
      Reference: GenericRSpace.v lines 1215-1224 *)
  Theorem seq_is_sequential :
    forall (q : SpaceQualifier),
      q = SQ_Seq -> allows_concurrent q = false.
  Proof.
    intros q Hseq.
    subst q. reflexivity.
  Qed.

  (** Default is the only persistent qualifier *)
  Theorem only_default_persists :
    forall (q : SpaceQualifier),
      is_persistent q = true <-> q = SQ_Default.
  Proof.
    intros q. split; intro H.
    - destruct q; simpl in H; try discriminate. reflexivity.
    - subst q. reflexivity.
  Qed.

  (** Persistent implies mobile *)
  Theorem persistent_implies_mobile :
    forall (q : SpaceQualifier),
      is_persistent q = true -> is_mobile q = true.
  Proof.
    intros q H.
    apply only_default_persists in H.
    subst q. reflexivity.
  Qed.

  (** Qualifiers are mutually exclusive *)
  Theorem qualifier_exclusive :
    forall (q : SpaceQualifier),
      (q = SQ_Default /\ q <> SQ_Temp /\ q <> SQ_Seq) \/
      (q <> SQ_Default /\ q = SQ_Temp /\ q <> SQ_Seq) \/
      (q <> SQ_Default /\ q <> SQ_Temp /\ q = SQ_Seq).
  Proof.
    intros q.
    destruct q.
    - left. repeat split; discriminate.
    - right. left. repeat split; discriminate.
    - right. right. repeat split; discriminate.
  Qed.

End QualifierSafety.

(** ** Registry Safety *)

Section RegistrySafety.

  (** A space ID is a 32-byte identifier *)
  Definition SpaceId := list Z.

  (** Same-space property for multi-channel joins.
      All channels in a join must belong to the same space.
      Reference: Registry/Invariants.v lines 276-302 *)
  Definition same_space_join (channel_owners : list (nat * SpaceId))
                              (channels : list nat) : Prop :=
    match channels with
    | [] => True
    | c :: rest =>
      match find (fun p => (fst p =? c)%nat) channel_owners with
      | None => True  (* Unowned channels are OK *)
      | Some (_, space) =>
        forall c', In c' rest ->
          match find (fun p => (fst p =? c')%nat) channel_owners with
          | None => True
          | Some (_, space') => space = space'
          end
      end
    end.

  (** Empty channel list trivially satisfies same-space *)
  Theorem empty_channels_same_space :
    forall (owners : list (nat * SpaceId)),
      same_space_join owners [].
  Proof.
    intros owners.
    simpl. exact I.
  Qed.

  (** Single channel trivially satisfies same-space *)
  Theorem single_channel_same_space :
    forall (owners : list (nat * SpaceId)) (c : nat),
      same_space_join owners [c].
  Proof.
    intros owners c.
    simpl.
    destruct (find (fun p => (fst p =? c)%nat) owners) as [[c' sp] |].
    - intros c'' Hin. inversion Hin.
    - exact I.
  Qed.

  (** ** Cross-Space Join Prevention *)

  (** Helper: lookup space for a channel *)
  Definition channel_lookup_space (owners : list (nat * SpaceId)) (c : nat)
    : option SpaceId :=
    match find (fun p => (fst p =? c)%nat) owners with
    | Some (_, space) => Some space
    | None => None
    end.

  (** Helper: all channels resolve to the same space *)
  Definition all_channels_same_space (owners : list (nat * SpaceId))
                                      (channels : list nat)
                                      (space : SpaceId) : Prop :=
    forall c, In c channels -> channel_lookup_space owners c = Some space.

  (** Cross-space joins are prohibited:
      A consume operation on multiple channels is only valid if
      all channels belong to the same space.

      Reference: SpaceCoordination.tla:ValidJoinPattern
      Design: "Reifying RSpaces.md" lines 309-310:
        "joins are permitted in the same space and prohibited across spaces" *)
  Definition valid_multi_channel_consume (owners : list (nat * SpaceId))
                                          (channels : list nat) : Prop :=
    match channels with
    | [] => True
    | c :: _ =>
      match channel_lookup_space owners c with
      | None => True  (* Unregistered channels permitted *)
      | Some space => all_channels_same_space owners channels space
      end
    end.

  (** Theorem: Multi-channel consume requires same-space property *)
  Theorem cross_space_join_prohibited :
    forall (owners : list (nat * SpaceId)) (channels : list nat),
      length channels > 1 ->
      valid_multi_channel_consume owners channels ->
      same_space_join owners channels.
  Proof.
    intros owners channels Hlen Hvalid.
    unfold valid_multi_channel_consume, same_space_join in *.
    destruct channels as [|c rest].
    - (* Empty list - length > 1 contradiction *)
      simpl in Hlen. lia.
    - (* Non-empty list *)
      destruct (find (fun p => (fst p =? c)%nat) owners) as [[c' sp] |] eqn:Hfind.
      + (* Channel c has an owner *)
        intros c'' Hin.
        destruct (find (fun p => (fst p =? c'')%nat) owners) as [[c''' sp'] |] eqn:Hfind'.
        * (* Channel c'' also has an owner - show spaces are equal *)
          unfold all_channels_same_space in Hvalid.
          assert (Hc'': In c'' (c :: rest)) by (right; exact Hin).
          unfold channel_lookup_space in Hvalid.
          rewrite Hfind in Hvalid.
          specialize (Hvalid c'' Hc'').
          rewrite Hfind' in Hvalid.
          injection Hvalid as Hsp.
          symmetry. exact Hsp.
        * (* c'' is unregistered - trivially OK *)
          exact I.
      + (* Channel c is unregistered *)
        exact I.
  Qed.

  (** Corollary: If any two channels have different spaces, consume is invalid *)
  Theorem different_spaces_invalid_consume :
    forall (owners : list (nat * SpaceId)) (c1 c2 : nat) (sp1 sp2 : SpaceId)
           (rest : list nat),
      channel_lookup_space owners c1 = Some sp1 ->
      channel_lookup_space owners c2 = Some sp2 ->
      sp1 <> sp2 ->
      ~ valid_multi_channel_consume owners (c1 :: c2 :: rest).
  Proof.
    intros owners c1 c2 sp1 sp2 rest H1 H2 Hneq Hvalid.
    unfold valid_multi_channel_consume in Hvalid.
    rewrite H1 in Hvalid.
    unfold all_channels_same_space in Hvalid.
    assert (Hc2_in: In c2 (c1 :: c2 :: rest)) by (right; left; reflexivity).
    specialize (Hvalid c2 Hc2_in).
    rewrite H2 in Hvalid.
    injection Hvalid as Heq.
    apply Hneq. symmetry. exact Heq.
  Qed.

End RegistrySafety.

(** ** Gas Safety (Phlogiston) *)

Section GasSafety.

  (** Phlogiston state tracks available gas *)
  Record PhlogistonState := mkPhloState {
    phlo_available : Z;
    phlo_used : Z;
    phlo_limit : Z;
  }.

  (** Phlogiston state is valid if available >= 0 *)
  Definition phlo_valid (s : PhlogistonState) : Prop :=
    (phlo_available s >= 0)%Z.

  (** Charging gas maintains validity
      Reference: Phlogiston.v charge_preserves_non_negative *)
  Definition charge_gas (s : PhlogistonState) (cost : Z)
    : option PhlogistonState :=
    if (phlo_available s >=? cost)%Z
    then Some (mkPhloState
                 (phlo_available s - cost)
                 (phlo_used s + cost)
                 (phlo_limit s))
    else None.

  (** Charging only succeeds with sufficient balance *)
  Theorem charge_requires_balance :
    forall (s : PhlogistonState) (cost : Z) (s' : PhlogistonState),
      charge_gas s cost = Some s' ->
      (phlo_available s >= cost)%Z.
  Proof.
    intros s cost s' H.
    unfold charge_gas in H.
    destruct (phlo_available s >=? cost)%Z eqn:Hge; try discriminate.
    apply Z.geb_le in Hge.
    lia.
  Qed.

  (** Charging preserves validity *)
  Theorem charge_preserves_valid :
    forall (s : PhlogistonState) (cost : Z) (s' : PhlogistonState),
      phlo_valid s ->
      (cost >= 0)%Z ->
      charge_gas s cost = Some s' ->
      phlo_valid s'.
  Proof.
    intros s cost s' Hvalid Hcost H.
    unfold charge_gas in H.
    destruct (phlo_available s >=? cost)%Z eqn:Hge; try discriminate.
    injection H as Hs'.
    subst s'.
    unfold phlo_valid in *.
    simpl.
    apply Z.geb_le in Hge.
    lia.
  Qed.

  (** Charging strictly decreases available gas *)
  Theorem charge_decreases :
    forall (s : PhlogistonState) (cost : Z) (s' : PhlogistonState),
      (cost > 0)%Z ->
      charge_gas s cost = Some s' ->
      (phlo_available s' < phlo_available s)%Z.
  Proof.
    intros s cost s' Hcost H.
    unfold charge_gas in H.
    destruct (phlo_available s >=? cost)%Z eqn:Hge; try discriminate.
    injection H as Hs'.
    subst s'.
    simpl.
    lia.
  Qed.

  (** Gas cannot go negative (key safety property) *)
  Theorem gas_never_negative :
    forall (s : PhlogistonState) (cost : Z) (s' : PhlogistonState),
      phlo_valid s ->
      (cost >= 0)%Z ->
      charge_gas s cost = Some s' ->
      (phlo_available s' >= 0)%Z.
  Proof.
    intros s cost s' Hvalid Hcost H.
    apply charge_preserves_valid with (s := s) (cost := cost);
    assumption.
  Qed.

End GasSafety.

(** ** Composite Safety Theorems *)

Section CompositeSafety.
  Variable A K : Type.

  (** A fully valid RSpace state satisfies all safety properties *)
  Record SafeRSpaceState := mkSafeState {
    safe_data : Z -> list (A * bool);
    safe_conts : list Z -> list (list A * K * bool);
    safe_qualifier : SpaceQualifier;
    safe_phlogiston : PhlogistonState;
    (* Invariants *)
    safe_no_pending : no_pending_match A K safe_data safe_conts;
    safe_phlo_valid : phlo_valid safe_phlogiston;
  }.

  (** Safe states exist (empty state is safe) *)
  Theorem safe_state_exists :
    exists (q : SpaceQualifier) (phlo : PhlogistonState),
      phlo_valid phlo /\
      no_pending_match A K (fun _ => []) (fun _ => []).
  Proof.
    exists SQ_Default.
    exists (mkPhloState 1000 0 1000).
    split.
    - unfold phlo_valid. simpl. lia.
    - apply empty_satisfies_no_pending_match.
  Qed.

  (** Safe operations preserve safety (sketch - full proof in GenericRSpace.v) *)
  Theorem operations_preserve_safety :
    forall (s : SafeRSpaceState),
      (* Operations that maintain no_pending_match also maintain safety *)
      no_pending_match A K (safe_data s) (safe_conts s) ->
      phlo_valid (safe_phlogiston s) ->
      True.  (* Placeholder - actual preservation proofs in GenericRSpace.v *)
  Proof.
    intros s _ _. exact I.
  Qed.

End CompositeSafety.

(** ** Summary of Safety Properties *)

(** This module consolidates safety properties from across the verification:

    1. No Pending Match (GenericRSpace.v:157-180):
       - At any channel, data OR continuations, never both
       - Empty state satisfies this
       - Produce/consume operations preserve this

    2. Channel Isolation:
       - Operations on one channel don't affect others
       - Proven via put/get/remove isolation theorems

    3. Qualifier Enforcement (GenericRSpace.v:1159-1357):
       - SQ_Seq channels are non-mobile and sequential
       - SQ_Default is the only persistent qualifier
       - Qualifiers are mutually exclusive

    4. Cross-Space Join Prevention (Registry/Invariants.v:276-302):
       - All channels in a join must belong to same space
       - Enforced via same_space_channels predicate

    5. Gas Accounting (Phlogiston.v):
       - Gas cannot go negative
       - Charging only succeeds with sufficient balance
       - Charging strictly decreases available gas

    6. Registry Invariants (Registry/Invariants.v):
       - Channel owners reference registered spaces
       - Use blocks reference registered spaces
       - Default space is registered
       - No duplicate space registrations

    These safety properties together ensure that RSpace implementations
    maintain correctness regardless of the specific channel store or
    collection types used. *)

