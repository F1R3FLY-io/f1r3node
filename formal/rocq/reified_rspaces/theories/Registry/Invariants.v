(** * Space Registry Invariants

    This module defines and proves invariants that the SpaceRegistry
    must maintain throughout its operation.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/registry.rs
*)

From Stdlib Require Import List Bool ZArith FMapList Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Helper: Channel Lookup Function *)

(** Look up the space ID for a channel hash in an ownership list *)
Definition channel_lookup (owners : list (nat * SpaceId)) (ch : nat)
  : option SpaceId :=
  match find (fun p => (fst p =? ch)%nat) owners with
  | Some p => Some (snd p)
  | None => None
  end.

(** ** Lemmas about find and lookup *)

Lemma find_some_In : forall A (f : A -> bool) l x,
  find f l = Some x -> In x l.
Proof.
  intros A f l.
  induction l as [| h t IH]; simpl; intros x Hfind.
  - discriminate.
  - destruct (f h) eqn:Hfh.
    + injection Hfind as Heq. subst. left. reflexivity.
    + right. apply IH. exact Hfind.
Qed.

Lemma find_some_satisfies : forall A (f : A -> bool) l x,
  find f l = Some x -> f x = true.
Proof.
  intros A f l.
  induction l as [| h t IH]; simpl; intros x Hfind.
  - discriminate.
  - destruct (f h) eqn:Hfh.
    + injection Hfind as Heq. subst. exact Hfh.
    + apply IH. exact Hfind.
Qed.

(** If channel is in owners, lookup returns the first matching space *)
Lemma In_channel_lookup : forall owners ch space,
  In (ch, space) owners ->
  exists sp, channel_lookup owners ch = Some sp.
Proof.
  intros owners ch space Hin.
  induction owners as [| [c s] t IH]; simpl in *.
  - contradiction.
  - destruct Hin as [Heq | Hin'].
    + injection Heq as Hc Hs. subst c s.
      unfold channel_lookup. simpl.
      rewrite Nat.eqb_refl.
      exists space. reflexivity.
    + unfold channel_lookup. simpl.
      destruct (c =? ch)%nat eqn:Hceq.
      * exists s. reflexivity.
      * specialize (IH Hin').
        destruct IH as [sp Hsp].
        unfold channel_lookup in Hsp.
        exists sp. exact Hsp.
Qed.

(** Uniqueness of channel keys: no duplicate channel hashes *)
Definition unique_channel_keys (owners : list (nat * SpaceId)) : Prop :=
  NoDup (map fst owners).

(** Under uniqueness, lookup returns exactly the space for a channel *)
Lemma lookup_unique : forall owners ch space,
  unique_channel_keys owners ->
  In (ch, space) owners ->
  channel_lookup owners ch = Some space.
Proof.
  intros owners ch space Huniq Hin.
  induction owners as [| [c s] t IH]; simpl in *.
  - contradiction.
  - destruct Hin as [Heq | Hin'].
    + injection Heq as Hc Hs. subst c s.
      unfold channel_lookup. simpl.
      rewrite Nat.eqb_refl.
      reflexivity.
    + unfold channel_lookup. simpl.
      destruct (c =? ch)%nat eqn:Hceq.
      * (* c = ch, but we have (ch, space) in tail - contradiction with uniqueness *)
        apply Nat.eqb_eq in Hceq. subst c.
        unfold unique_channel_keys in Huniq.
        simpl in Huniq.
        inversion Huniq as [| x l Hnotin Hrest Heql]. subst.
        exfalso.
        apply Hnotin.
        apply in_map_iff.
        exists (ch, space).
        split; [reflexivity | exact Hin'].
      * (* c <> ch, recurse *)
        unfold unique_channel_keys in Huniq.
        simpl in Huniq.
        inversion Huniq as [| x l Hnotin Hrest Heql]. subst.
        unfold unique_channel_keys in IH.
        specialize (IH Hrest Hin').
        unfold channel_lookup in IH.
        exact IH.
Qed.

(** ** Registry State *)

(** Represents a registered space entry *)
Record SpaceEntry := mkSpaceEntry {
  se_id : SpaceId;
  se_config : SpaceConfig;
  se_active : bool;
}.

(** Registry state *)
Record RegistryState := mkRegistry {
  rs_spaces : list SpaceEntry;
  rs_channel_owners : list (nat * SpaceId);  (* channel_hash -> space_id *)
  rs_use_block_stack : UseBlockStack;
  rs_default_space : option SpaceId;
}.

(** Empty registry *)
Definition empty_registry : RegistryState :=
  mkRegistry [] [] [] None.

(** ** Invariants *)

(** All channel owners reference registered spaces *)
Definition inv_channel_owners_valid (rs : RegistryState) : Prop :=
  forall ch_hash space_id,
    In (ch_hash, space_id) (rs_channel_owners rs) ->
    exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id.

(** Use block stack only contains registered spaces *)
Definition inv_use_blocks_valid (rs : RegistryState) : Prop :=
  forall space_id,
    In space_id (rs_use_block_stack rs) ->
    exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id.

(** Default space, if set, is registered *)
Definition inv_default_space_valid (rs : RegistryState) : Prop :=
  match rs_default_space rs with
  | None => True
  | Some space_id =>
      exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id
  end.

(** No duplicate space IDs *)
Definition inv_no_duplicate_spaces (rs : RegistryState) : Prop :=
  NoDup (map se_id (rs_spaces rs)).

(** No duplicate channel hashes in ownership *)
Definition inv_unique_channel_ownership (rs : RegistryState) : Prop :=
  unique_channel_keys (rs_channel_owners rs).

(** Combined invariant *)
Definition registry_invariant (rs : RegistryState) : Prop :=
  inv_channel_owners_valid rs /\
  inv_use_blocks_valid rs /\
  inv_default_space_valid rs /\
  inv_no_duplicate_spaces rs /\
  inv_unique_channel_ownership rs.

(** ** Invariant Preservation Theorems *)

(** Empty registry satisfies invariant *)
Theorem empty_registry_satisfies_invariant :
  registry_invariant empty_registry.
Proof.
  unfold registry_invariant, empty_registry.
  split; [| split; [| split; [| split]]].
  - unfold inv_channel_owners_valid. simpl. intros ch_hash space_id H. contradiction.
  - unfold inv_use_blocks_valid. simpl. intros space_id H. contradiction.
  - unfold inv_default_space_valid. simpl. trivial.
  - unfold inv_no_duplicate_spaces. simpl. constructor.
  - unfold inv_unique_channel_ownership, unique_channel_keys. simpl. constructor.
Qed.

(** Register space preserves invariant (when ID is fresh) *)
Definition register_space (rs : RegistryState) (entry : SpaceEntry)
  : RegistryState :=
  mkRegistry
    (entry :: rs_spaces rs)
    (rs_channel_owners rs)
    (rs_use_block_stack rs)
    (rs_default_space rs).

Theorem register_space_preserves_invariant :
  forall rs entry,
    registry_invariant rs ->
    ~ In (se_id entry) (map se_id (rs_spaces rs)) ->
    registry_invariant (register_space rs entry).
Proof.
  intros rs entry Hinv Hfresh.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, register_space.
  split; [| split; [| split; [| split]]].
  - (* Channel owners still valid *)
    unfold inv_channel_owners_valid in *.
    simpl. intros ch_hash space_id Hin.
    specialize (Hco ch_hash space_id Hin).
    destruct Hco as [e [Hin' Heq]].
    exists e. split; [right; exact Hin' | exact Heq].
  - (* Use blocks still valid *)
    unfold inv_use_blocks_valid in *.
    simpl. intros space_id Hin.
    specialize (Hub space_id Hin).
    destruct Hub as [e [Hin' Heq]].
    exists e. split; [right; exact Hin' | exact Heq].
  - (* Default space still valid *)
    unfold inv_default_space_valid in *.
    simpl. destruct (rs_default_space rs) eqn:Hds.
    + destruct Hdef as [e [Hin' Heq]].
      exists e. split; [right; exact Hin' | exact Heq].
    + trivial.
  - (* No duplicates preserved *)
    unfold inv_no_duplicate_spaces in *.
    simpl. constructor.
    + exact Hfresh.
    + exact Hnd.
  - (* Channel uniqueness preserved - channel owners unchanged *)
    exact Huniq.
Qed.

(** ** Cross-Space Join Prevention *)

(** Channels in a join must belong to the same space.
    Uses the module-level channel_lookup function. *)
Definition same_space_channels (rs : RegistryState) (channels : list nat) : Prop :=
  match channels with
  | [] => True
  | ch :: rest =>
      let first_space := channel_lookup (rs_channel_owners rs) ch in
      forall ch', In ch' rest -> channel_lookup (rs_channel_owners rs) ch' = first_space
  end.

(** Helper: if two channels are both in the join pattern,
    they have the same lookup result *)
Lemma same_space_channels_eq : forall rs channels ch1 ch2,
  In ch1 channels ->
  In ch2 channels ->
  same_space_channels rs channels ->
  channel_lookup (rs_channel_owners rs) ch1 = channel_lookup (rs_channel_owners rs) ch2.
Proof.
  intros rs channels ch1 ch2 Hin1 Hin2 Hsame.
  destruct channels as [| first rest].
  - (* Empty list - contradiction *)
    contradiction.
  - (* Non-empty list *)
    simpl in Hsame.
    destruct Hin1 as [Heq1 | Hin1'].
    + (* ch1 is the first element *)
      subst ch1.
      destruct Hin2 as [Heq2 | Hin2'].
      * (* ch2 is also the first element *)
        subst ch2. reflexivity.
      * (* ch2 is in rest *)
        symmetry. apply Hsame. exact Hin2'.
    + (* ch1 is in rest *)
      destruct Hin2 as [Heq2 | Hin2'].
      * (* ch2 is the first element *)
        subst ch2. apply Hsame. exact Hin1'.
      * (* both in rest *)
        transitivity (channel_lookup (rs_channel_owners rs) first).
        -- apply Hsame. exact Hin1'.
        -- symmetry. apply Hsame. exact Hin2'.
Qed.

(** Cross-space joins are disallowed.
    Requires that channel ownership is unique (part of registry invariant). *)
Theorem cross_space_join_forbidden :
  forall rs channels space1 space2 ch1 ch2,
    inv_unique_channel_ownership rs ->
    In (ch1, space1) (rs_channel_owners rs) ->
    In (ch2, space2) (rs_channel_owners rs) ->
    In ch1 channels ->
    In ch2 channels ->
    same_space_channels rs channels ->
    space1 = space2.
Proof.
  intros rs channels space1 space2 ch1 ch2 Huniq Hin1 Hin2 Hch1 Hch2 Hsame.
  unfold inv_unique_channel_ownership in Huniq.
  (* Use lookup_unique to get the lookup results *)
  assert (Hlookup1 : channel_lookup (rs_channel_owners rs) ch1 = Some space1).
  { apply lookup_unique; assumption. }
  assert (Hlookup2 : channel_lookup (rs_channel_owners rs) ch2 = Some space2).
  { apply lookup_unique; assumption. }
  (* Both lookups are equal by same_space_channels *)
  assert (Heq : channel_lookup (rs_channel_owners rs) ch1 =
                channel_lookup (rs_channel_owners rs) ch2).
  { apply same_space_channels_eq with channels; assumption. }
  (* Conclude space1 = space2 *)
  rewrite Hlookup1 in Heq.
  rewrite Hlookup2 in Heq.
  injection Heq as Heq'.
  exact Heq'.
Qed.

(** ** Additional Registry Operations *)

(** Boolean equality for SpaceId (list Z) *)
Fixpoint beq_space_id (s1 s2 : SpaceId) : bool :=
  match s1, s2 with
  | [], [] => true
  | x :: xs, y :: ys => Z.eqb x y && beq_space_id xs ys
  | _, _ => false
  end.

(** Helper: beq_space_id reflects equality *)
Lemma beq_space_id_eq : forall s1 s2,
  beq_space_id s1 s2 = true <-> s1 = s2.
Proof.
  intros s1.
  induction s1 as [| x xs IH]; intros s2; split; intro H.
  - (* [] -> s2, beq = true -> eq *)
    destruct s2 as [| y ys].
    + reflexivity.
    + simpl in H. discriminate.
  - (* [] -> s2, eq -> beq = true *)
    subst s2. reflexivity.
  - (* x::xs -> s2, beq = true -> eq *)
    destruct s2 as [| y ys].
    + simpl in H. discriminate.
    + simpl in H.
      apply andb_prop in H.
      destruct H as [Hxy Hrest].
      apply Z.eqb_eq in Hxy.
      apply IH in Hrest.
      subst x. f_equal. exact Hrest.
  - (* x::xs -> s2, eq -> beq = true *)
    subst s2. simpl.
    rewrite Z.eqb_refl.
    apply IH. reflexivity.
Qed.

(** Unregister a space by ID (marks as inactive, doesn't remove) *)
Definition unregister_space (rs : RegistryState) (space_id : SpaceId)
  : RegistryState :=
  let spaces' := map (fun e =>
    if beq_space_id (se_id e) space_id
    then mkSpaceEntry (se_id e) (se_config e) false
    else e) (rs_spaces rs) in
  mkRegistry
    spaces'
    (rs_channel_owners rs)
    (rs_use_block_stack rs)
    (rs_default_space rs).

(** Helper: map preserves In with transformation *)
Lemma In_map_transform : forall A B (f : A -> B) (l : list A) x,
  In x l -> In (f x) (map f l).
Proof.
  intros A B f l.
  induction l as [| h t IH]; simpl; intros x Hin.
  - contradiction.
  - destruct Hin as [Heq | Hin'].
    + left. f_equal. exact Heq.
    + right. apply IH. exact Hin'.
Qed.

(** Helper: unregister preserves space IDs *)
Lemma unregister_preserves_ids : forall rs space_id,
  map se_id (rs_spaces (unregister_space rs space_id)) = map se_id (rs_spaces rs).
Proof.
  intros rs space_id.
  unfold unregister_space. simpl.
  induction (rs_spaces rs) as [| e t IH].
  - reflexivity.
  - simpl.
    destruct (beq_space_id (se_id e) space_id) eqn:Heq.
    + simpl. f_equal. exact IH.
    + simpl. f_equal. exact IH.
Qed.

(** Unregister space preserves invariant *)
Theorem unregister_preserves_invariant :
  forall rs space_id,
    registry_invariant rs ->
    registry_invariant (unregister_space rs space_id).
Proof.
  intros rs space_id Hinv.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, unregister_space.
  split; [| split; [| split; [| split]]].
  - (* Channel owners still valid *)
    unfold inv_channel_owners_valid in *.
    simpl. intros ch_hash sid Hin.
    specialize (Hco ch_hash sid Hin).
    destruct Hco as [e [Hin' Heq]].
    (* e is in rs_spaces, need to find corresponding entry in mapped list *)
    exists (if beq_space_id (se_id e) space_id
            then mkSpaceEntry (se_id e) (se_config e) false
            else e).
    split.
    + apply in_map_iff.
      exists e. split.
      * reflexivity.
      * exact Hin'.
    + destruct (beq_space_id (se_id e) space_id); simpl; exact Heq.
  - (* Use blocks still valid *)
    unfold inv_use_blocks_valid in *.
    simpl. intros sid Hin.
    specialize (Hub sid Hin).
    destruct Hub as [e [Hin' Heq]].
    exists (if beq_space_id (se_id e) space_id
            then mkSpaceEntry (se_id e) (se_config e) false
            else e).
    split.
    + apply in_map_iff.
      exists e. split; [reflexivity | exact Hin'].
    + destruct (beq_space_id (se_id e) space_id); simpl; exact Heq.
  - (* Default space still valid *)
    unfold inv_default_space_valid in *.
    simpl. destruct (rs_default_space rs) eqn:Hds.
    + destruct Hdef as [e [Hin' Heq]].
      exists (if beq_space_id (se_id e) space_id
              then mkSpaceEntry (se_id e) (se_config e) false
              else e).
      split.
      * apply in_map_iff.
        exists e. split; [reflexivity | exact Hin'].
      * destruct (beq_space_id (se_id e) space_id); simpl; exact Heq.
    + trivial.
  - (* No duplicates preserved *)
    unfold inv_no_duplicate_spaces in *.
    simpl.
    (* Goal: NoDup (map se_id (map f (rs_spaces rs))) where f preserves se_id *)
    (* Prove that map se_id (map f l) = map se_id l when f preserves se_id *)
    assert (Hmap: map se_id (map (fun e : SpaceEntry =>
       if beq_space_id (se_id e) space_id
       then {| se_id := se_id e; se_config := se_config e; se_active := false |}
       else e) (rs_spaces rs)) = map se_id (rs_spaces rs)).
    { clear Hnd Huniq Hco Hub Hdef.
      induction (rs_spaces rs) as [| e t IH].
      - reflexivity.
      - simpl.
        destruct (beq_space_id (se_id e) space_id) eqn:Heq.
        + simpl. f_equal. exact IH.
        + simpl. f_equal. exact IH.
    }
    rewrite Hmap.
    exact Hnd.
  - (* Channel uniqueness preserved - channel owners unchanged *)
    exact Huniq.
Qed.

(** Push space onto use block stack *)
Definition use_block_push (rs : RegistryState) (space_id : SpaceId)
  : RegistryState :=
  mkRegistry
    (rs_spaces rs)
    (rs_channel_owners rs)
    (space_id :: rs_use_block_stack rs)
    (rs_default_space rs).

(** Use block push preserves invariant (when space is registered) *)
Theorem use_block_push_preserves_invariant :
  forall rs space_id,
    registry_invariant rs ->
    (exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id) ->
    registry_invariant (use_block_push rs space_id).
Proof.
  intros rs space_id Hinv Hreg.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, use_block_push.
  split; [| split; [| split; [| split]]].
  - (* Channel owners unchanged *)
    simpl. exact Hco.
  - (* Use blocks still valid *)
    unfold inv_use_blocks_valid in *.
    simpl. intros sid Hin.
    destruct Hin as [Heq | Hin'].
    + (* The pushed space_id *)
      subst sid. exact Hreg.
    + (* Already in stack *)
      apply Hub. exact Hin'.
  - (* Default space unchanged *)
    simpl. exact Hdef.
  - (* No duplicates unchanged *)
    simpl. exact Hnd.
  - (* Channel uniqueness unchanged *)
    exact Huniq.
Qed.

(** Pop space from use block stack *)
Definition use_block_pop (rs : RegistryState) : RegistryState :=
  mkRegistry
    (rs_spaces rs)
    (rs_channel_owners rs)
    (match rs_use_block_stack rs with
     | [] => []
     | _ :: rest => rest
     end)
    (rs_default_space rs).

(** Use block pop preserves invariant *)
Theorem use_block_pop_preserves_invariant :
  forall rs,
    registry_invariant rs ->
    registry_invariant (use_block_pop rs).
Proof.
  intros rs Hinv.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, use_block_pop.
  split; [| split; [| split; [| split]]].
  - (* Channel owners unchanged *)
    simpl. exact Hco.
  - (* Use blocks still valid - subset of original *)
    unfold inv_use_blocks_valid in *.
    simpl. intros sid Hin.
    destruct (rs_use_block_stack rs) as [| top rest] eqn:Hstack.
    + (* Empty stack - contradiction *)
      contradiction.
    + (* Non-empty stack - sid in rest *)
      apply Hub. right. exact Hin.
  - (* Default space unchanged *)
    simpl. exact Hdef.
  - (* No duplicates unchanged *)
    simpl. exact Hnd.
  - (* Channel uniqueness unchanged *)
    exact Huniq.
Qed.

(** Add channel ownership *)
Definition add_channel_owner (rs : RegistryState) (ch_hash : nat) (space_id : SpaceId)
  : RegistryState :=
  mkRegistry
    (rs_spaces rs)
    ((ch_hash, space_id) :: rs_channel_owners rs)
    (rs_use_block_stack rs)
    (rs_default_space rs).

(** Add channel ownership preserves invariant (when space is registered and channel is fresh) *)
Theorem add_channel_owner_preserves_invariant :
  forall rs ch_hash space_id,
    registry_invariant rs ->
    (exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id) ->
    ~ In ch_hash (map fst (rs_channel_owners rs)) ->
    registry_invariant (add_channel_owner rs ch_hash space_id).
Proof.
  intros rs ch_hash space_id Hinv Hreg Hfresh.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, add_channel_owner.
  split; [| split; [| split; [| split]]].
  - (* Channel owners valid *)
    unfold inv_channel_owners_valid in *.
    simpl. intros ch sid Hin.
    destruct Hin as [Heq | Hin'].
    + (* The new channel *)
      injection Heq as Hch Hsid. subst ch sid.
      exact Hreg.
    + (* Already in owners *)
      apply (Hco ch sid). exact Hin'.
  - (* Use blocks unchanged *)
    simpl. exact Hub.
  - (* Default space unchanged *)
    simpl. exact Hdef.
  - (* No duplicates unchanged *)
    simpl. exact Hnd.
  - (* Channel uniqueness preserved *)
    unfold inv_unique_channel_ownership, unique_channel_keys in *.
    simpl. constructor.
    + exact Hfresh.
    + exact Huniq.
Qed.

(** Set default space *)
Definition set_default_space (rs : RegistryState) (space_id : SpaceId)
  : RegistryState :=
  mkRegistry
    (rs_spaces rs)
    (rs_channel_owners rs)
    (rs_use_block_stack rs)
    (Some space_id).

(** Set default space preserves invariant (when space is registered) *)
Theorem set_default_space_preserves_invariant :
  forall rs space_id,
    registry_invariant rs ->
    (exists entry, In entry (rs_spaces rs) /\ se_id entry = space_id) ->
    registry_invariant (set_default_space rs space_id).
Proof.
  intros rs space_id Hinv Hreg.
  destruct Hinv as [Hco [Hub [Hdef [Hnd Huniq]]]].
  unfold registry_invariant, set_default_space.
  split; [| split; [| split; [| split]]].
  - (* Channel owners unchanged *)
    simpl. exact Hco.
  - (* Use blocks unchanged *)
    simpl. exact Hub.
  - (* Default space now valid *)
    unfold inv_default_space_valid.
    simpl. exact Hreg.
  - (* No duplicates unchanged *)
    simpl. exact Hnd.
  - (* Channel uniqueness unchanged *)
    exact Huniq.
Qed.
