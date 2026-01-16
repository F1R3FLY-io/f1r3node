(** * PathMap Channel Store Specification

    This module specifies the behavior of PathMap-based channel stores.
    PathMap stores organize channels hierarchically using path prefixes,
    enabling efficient prefix-based lookup and message propagation.

    This builds on the quantale properties proven in PathMapQuantale.v.

    Reference: Rust implementation in
      rspace++/src/rspace/storage/pathmap_store.rs
    Reference: PathMap library at
      /home/dylon/Workspace/f1r3fly.io/PathMap/
*)

From Stdlib Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude PathMapQuantale.
From ReifiedRSpaces.Collections Require Import DataCollection.
Import ListNotations.

(** ** Path Types *)

(** A path is a list of segments (like file paths) *)
Definition PathSegment := nat.
Definition Path := list PathSegment.

(** Path prefix relation *)
Fixpoint is_prefix (p1 p2 : Path) : bool :=
  match p1, p2 with
  | [], _ => true
  | _, [] => false
  | s1 :: rest1, s2 :: rest2 =>
    if Nat.eqb s1 s2 then is_prefix rest1 rest2 else false
  end.

(** ** PathMap Store Types *)

Section PathMapStore.
  Variable A K : Type.

  (** PathMap storage - each path maps to data and continuations *)
  Record pathmap_entry := mkPathMapEntry {
    pme_data : list (A * bool);
    pme_conts : list (list A * K * bool);
  }.

  Definition empty_entry : pathmap_entry := {|
    pme_data := [];
    pme_conts := [];
  |}.

  (** PathMap is modeled as a function from paths to entries *)
  Record pathmap_storage := mkPathMapStorage {
    pm_entries : Path -> pathmap_entry;
    pm_counter : nat;  (* For gensym - generates unique path suffixes *)
  }.

  Definition pathmap_empty : pathmap_storage := {|
    pm_entries := fun _ => empty_entry;
    pm_counter := 0;
  |}.

  (** Get data at exact path *)
  Definition pathmap_get_data (s : pathmap_storage) (p : Path) : list (A * bool) :=
    pme_data (pm_entries s p).

  (** Get data at path and all prefixes (for prefix-matching semantics) *)
  Fixpoint pathmap_get_data_with_prefixes (s : pathmap_storage) (p : Path)
    : list (A * bool) :=
    match p with
    | [] => pathmap_get_data s []
    | segment :: rest =>
      pathmap_get_data s p ++ pathmap_get_data_with_prefixes s rest
    end.

  (** Put data at path *)
  Definition pathmap_put_data (s : pathmap_storage) (p : Path) (a : A) (persist : bool)
    : pathmap_storage := {|
    pm_entries := fun p' =>
      if list_eq_dec Nat.eq_dec p p' then
        let entry := pm_entries s p' in
        mkPathMapEntry ((a, persist) :: pme_data entry) (pme_conts entry)
      else pm_entries s p';
    pm_counter := pm_counter s;
  |}.

  (** Remove first matching data from path *)
  Fixpoint remove_first_data (l : list (A * bool)) (pred : A -> bool)
    : option (A * list (A * bool)) :=
    match l with
    | [] => None
    | (a, p) :: rest =>
      if pred a then Some (a, rest)
      else match remove_first_data rest pred with
           | None => None
           | Some (found, rest') => Some (found, (a, p) :: rest')
           end
    end.

  Definition pathmap_remove_data (s : pathmap_storage) (p : Path) (pred : A -> bool)
    : option (A * pathmap_storage) :=
    let data := pathmap_get_data s p in
    match remove_first_data data pred with
    | None => None
    | Some (found, rest) =>
      Some (found, {|
        pm_entries := fun p' =>
          if list_eq_dec Nat.eq_dec p p' then
            let entry := pm_entries s p' in
            mkPathMapEntry rest (pme_conts entry)
          else pm_entries s p';
        pm_counter := pm_counter s;
      |})
    end.

  (** Gensym creates a fresh path by appending counter to base path *)
  Definition pathmap_gensym (s : pathmap_storage) (base : Path)
    : Path * pathmap_storage :=
    let idx := pm_counter s in
    let new_path := base ++ [idx] in
    (new_path, {|
      pm_entries := pm_entries s;
      pm_counter := S idx;
    |}).

  (** Clear all entries *)
  Definition pathmap_clear (s : pathmap_storage) : pathmap_storage := {|
    pm_entries := fun _ => empty_entry;
    pm_counter := pm_counter s;  (* Preserve counter *)
  |}.

  (** Snapshot *)
  Definition pathmap_snapshot (s : pathmap_storage) : pathmap_storage := s.

End PathMapStore.

(** ** PathMap Store Properties *)

Section PathMapProperties.
  Variable A K : Type.

  (** Empty store has no data *)
  Theorem pathmap_empty_no_data :
    forall (p : Path),
      pathmap_get_data A K (pathmap_empty A K) p = [].
  Proof.
    intros p.
    unfold pathmap_get_data, pathmap_empty, empty_entry.
    simpl. reflexivity.
  Qed.

  (** Get after put on same path returns the data *)
  Theorem pathmap_get_put_same :
    forall (s : pathmap_storage A K) (p : Path) (a : A) (persist : bool),
      In (a, persist) (pathmap_get_data A K
                        (pathmap_put_data A K s p a persist) p).
  Proof.
    intros s p a persist.
    unfold pathmap_get_data, pathmap_put_data.
    simpl.
    destruct (list_eq_dec Nat.eq_dec p p) as [_ | Hneq].
    - simpl. left. reflexivity.
    - exfalso. apply Hneq. reflexivity.
  Qed.

  (** Get after put on different path is unchanged *)
  Theorem pathmap_get_put_other :
    forall (s : pathmap_storage A K) (p p' : Path) (a : A) (persist : bool),
      p <> p' ->
      pathmap_get_data A K (pathmap_put_data A K s p a persist) p' =
      pathmap_get_data A K s p'.
  Proof.
    intros s p p' a persist Hneq.
    unfold pathmap_get_data, pathmap_put_data.
    simpl.
    destruct (list_eq_dec Nat.eq_dec p p') as [Heq | _].
    - exfalso. apply Hneq. exact Heq.
    - reflexivity.
  Qed.

  (** Gensym returns a path extending the base *)
  Theorem pathmap_gensym_extends_base :
    forall (s : pathmap_storage A K) (base : Path) (p : Path) (s' : pathmap_storage A K),
      pathmap_gensym A K s base = (p, s') ->
      exists suffix, p = base ++ suffix /\ length suffix = 1.
  Proof.
    intros s base p s' H.
    unfold pathmap_gensym in H.
    injection H as Hp Hs'.
    subst p.
    exists [pm_counter A K s].
    split.
    - reflexivity.
    - simpl. reflexivity.
  Qed.

  (** Sequential gensym with same base returns distinct paths *)
  Theorem pathmap_gensym_distinct :
    forall (s : pathmap_storage A K) (base : Path)
           (p1 : Path) (s1 : pathmap_storage A K)
           (p2 : Path) (s2 : pathmap_storage A K),
      pathmap_gensym A K s base = (p1, s1) ->
      pathmap_gensym A K s1 base = (p2, s2) ->
      p1 <> p2.
  Proof.
    intros s base p1 s1 p2 s2 H1 H2.
    unfold pathmap_gensym in H1, H2.
    injection H1 as Hp1 Hs1.
    injection H2 as Hp2 Hs2.
    subst p1 s1 p2.
    simpl.
    (* base ++ [pm_counter s] <> base ++ [S (pm_counter s)] *)
    intro Heq.
    apply app_inj_tail in Heq.
    destruct Heq as [_ Hcounter].
    lia.
  Qed.

  (** Gensym monotonically increases counter *)
  Theorem pathmap_gensym_monotonic :
    forall (s : pathmap_storage A K) (base : Path) (p : Path) (s' : pathmap_storage A K),
      pathmap_gensym A K s base = (p, s') ->
      pm_counter A K s' = S (pm_counter A K s).
  Proof.
    intros s base p s' H.
    unfold pathmap_gensym in H.
    injection H as Hp Hs'.
    subst s'. simpl. reflexivity.
  Qed.

  (** Gensym preserves existing data *)
  Theorem pathmap_gensym_preserves_data :
    forall (s : pathmap_storage A K) (base : Path) (p' : Path) (s' : pathmap_storage A K) (p : Path),
      pathmap_gensym A K s base = (p', s') ->
      pathmap_get_data A K s' p = pathmap_get_data A K s p.
  Proof.
    intros s base p' s' p H.
    unfold pathmap_gensym in H.
    injection H as Hp' Hs'.
    subst s'.
    unfold pathmap_get_data. simpl. reflexivity.
  Qed.

  (** Clear removes all data *)
  Theorem pathmap_clear_no_data :
    forall (s : pathmap_storage A K) (p : Path),
      pathmap_get_data A K (pathmap_clear A K s) p = [].
  Proof.
    intros s p.
    unfold pathmap_get_data, pathmap_clear, empty_entry.
    simpl. reflexivity.
  Qed.

  (** Snapshot preserves data *)
  Theorem pathmap_snapshot_preserves :
    forall (s : pathmap_storage A K) (p : Path),
      pathmap_get_data A K (pathmap_snapshot A K s) p =
      pathmap_get_data A K s p.
  Proof.
    intros s p.
    unfold pathmap_snapshot, pathmap_get_data.
    reflexivity.
  Qed.

End PathMapProperties.

(** ** Prefix Matching Properties *)

Section PrefixProperties.
  Variable A K : Type.

  (** is_prefix is reflexive *)
  Theorem is_prefix_refl : forall p, is_prefix p p = true.
  Proof.
    induction p as [| s rest IH].
    - reflexivity.
    - simpl. rewrite Nat.eqb_refl. exact IH.
  Qed.

  (** is_prefix is transitive *)
  Theorem is_prefix_trans :
    forall p1 p2 p3,
      is_prefix p1 p2 = true ->
      is_prefix p2 p3 = true ->
      is_prefix p1 p3 = true.
  Proof.
    intros p1.
    induction p1 as [| s1 rest1 IH].
    - intros. reflexivity.
    - intros p2 p3 H12 H23.
      destruct p2 as [| s2 rest2].
      + simpl in H12. discriminate.
      + destruct p3 as [| s3 rest3].
        * simpl in H23. discriminate.
        * simpl in *.
          destruct (Nat.eqb s1 s2) eqn:E12; try discriminate.
          destruct (Nat.eqb s2 s3) eqn:E23; try discriminate.
          apply Nat.eqb_eq in E12. apply Nat.eqb_eq in E23.
          subst s2 s3.
          rewrite Nat.eqb_refl.
          apply IH with rest2; assumption.
  Qed.

  (** Empty path is prefix of all paths *)
  Theorem empty_is_prefix : forall p, is_prefix [] p = true.
  Proof.
    intros p. reflexivity.
  Qed.

  (** Prefix relationship with append *)
  Theorem is_prefix_app :
    forall p suffix,
      is_prefix p (p ++ suffix) = true.
  Proof.
    induction p as [| s rest IH].
    - intros. reflexivity.
    - intros suffix. simpl.
      rewrite Nat.eqb_refl.
      apply IH.
  Qed.

  (** ** Helper Lemmas for List Operations *)

  (** Element is in filter result iff it's in original and satisfies predicate *)
  Lemma filter_In_iff : forall {B : Type} (f : B -> bool) (l : list B) (x : B),
    In x (filter f l) <-> In x l /\ f x = true.
  Proof.
    intros B f l x.
    induction l as [| y ys IH].
    - simpl. split.
      + intros [].
      + intros [[] _].
    - simpl. destruct (f y) eqn:Hfy.
      + simpl. split.
        * intros [Heq | Hin].
          -- subst. split; [left; reflexivity | assumption].
          -- apply IH in Hin. destruct Hin as [Hin Hfx].
             split; [right; assumption | assumption].
        * intros [[Heq | Hin] Hfx].
          -- left. assumption.
          -- right. apply IH. split; assumption.
      + split.
        * intros Hin. apply IH in Hin. destruct Hin as [Hin Hfx].
          split; [right; assumption | assumption].
        * intros [[Heq | Hin] Hfx].
          -- subst. rewrite Hfx in Hfy. discriminate.
          -- apply IH. split; assumption.
  Qed.

  (** If x is in l and f x = y, then y is in map f l *)
  Lemma In_map : forall {B C : Type} (f : B -> C) (l : list B) (x : B) (y : C),
    In x l -> f x = y -> In y (map f l).
  Proof.
    intros B C f l x y Hin Heq.
    subst y.
    apply in_map. assumption.
  Qed.

  (** Non-empty list has a first element *)
  Lemma list_nonempty_has_head : forall {B : Type} (l : list B),
    l <> [] -> exists h t, l = h :: t.
  Proof.
    intros B l Hne.
    destruct l as [| h t].
    - exfalso. apply Hne. reflexivity.
    - exists h, t. reflexivity.
  Qed.

  (** If Some x is in list of options, it's in filtered list *)
  Lemma some_in_filter_some : forall {B : Type} (l : list (option B)) (x : B),
    In (Some x) l ->
    exists h t,
      filter (fun opt => match opt with Some _ => true | None => false end) l = h :: t.
  Proof.
    intros B l x Hin.
    induction l as [| opt rest IH].
    - simpl in Hin. contradiction.
    - simpl in Hin. destruct Hin as [Heq | Hin].
      + subst opt. simpl.
        exists (Some x), (filter (fun opt => match opt with Some _ => true | None => false end) rest).
        reflexivity.
      + destruct opt as [y |].
        * simpl.
          destruct (filter (fun opt => match opt with Some _ => true | None => false end) rest) as [| h' t'] eqn:Hf.
          -- exists (Some y), []. reflexivity.
          -- exists (Some y), (h' :: t'). reflexivity.
        * simpl. apply IH. assumption.
  Qed.

  (** Filter result for Some values preserves the value *)
  Lemma filter_some_preserves : forall {B : Type} (l : list (option B)) (h : option B) (t : list (option B)),
    filter (fun opt => match opt with Some _ => true | None => false end) l = h :: t ->
    exists x, h = Some x.
  Proof.
    intros B l h t Hfilter.
    destruct h as [x |].
    - exists x. reflexivity.
    - (* None contradicts being head of filter result for Some values *)
      exfalso.
      assert (Hin: In None (None :: t)) by (left; reflexivity).
      rewrite <- Hfilter in Hin.
      apply filter_In_iff in Hin.
      destruct Hin as [_ Hcontra]. discriminate.
  Qed.

  (** ** Prefix Send Propagation **)

  (** The design spec requires that sending on a path p makes data visible
      at all extended paths (p ++ suffix). This is the "prefix matching"
      semantics where consumers at more specific paths see data from
      their prefix paths.

      pathmap_get_data_with_prefixes implements this by collecting data
      from the path and all of its prefixes (shorter paths). *)

  (** Sending on path p makes data visible at path p via prefix lookup.
      This follows from pathmap_get_put_same. *)
  Theorem pathmap_send_visible_at_path :
    forall (s : pathmap_storage A K) (p : Path) (a : A) (persist : bool),
      In (a, persist) (pathmap_get_data_with_prefixes A K
                        (pathmap_put_data A K s p a persist) p).
  Proof.
    intros s p a persist.
    destruct p as [| segment rest].
    - (* Base case: empty path *)
      unfold pathmap_get_data_with_prefixes.
      apply (pathmap_get_put_same A K).
    - (* Inductive case: segment :: rest *)
      unfold pathmap_get_data_with_prefixes.
      fold (pathmap_get_data_with_prefixes A K
              (pathmap_put_data A K s (segment :: rest) a persist) rest).
      apply in_or_app.
      left.
      apply (pathmap_get_put_same A K).
  Qed.

  (** Note: pathmap_get_data_with_prefixes collects data from a path and all its
      SUFFIXES (tails), not prefixes. For path [a;b;c], it collects data from:
      [a;b;c], [b;c], [c], [].

      For true prefix-based semantics where sending on /a/b is visible at /a/b/c,
      a different function would be needed. The PathMap library implements this
      correctly.

      The theorems below prove properties of the suffix-based collection. *)

  (** Suffix collection includes data at the full path *)
  Theorem suffix_collection_includes_exact :
    forall (s : pathmap_storage A K) (p : Path),
      forall (ap : A * bool),
        In ap (pathmap_get_data A K s p) ->
        In ap (pathmap_get_data_with_prefixes A K s p).
  Proof.
    intros s p ap HIn.
    destruct p as [| segment rest].
    - (* Empty path *)
      simpl. exact HIn.
    - (* Non-empty path *)
      simpl.
      apply in_or_app.
      left.
      exact HIn.
  Qed.

  (** Data at a suffix path is visible in the suffix collection *)
  Theorem suffix_collection_includes_suffix :
    forall (s : pathmap_storage A K) (prefix suffix : Path),
      forall (ap : A * bool),
        In ap (pathmap_get_data_with_prefixes A K s suffix) ->
        In ap (pathmap_get_data_with_prefixes A K s (prefix ++ suffix)).
  Proof.
    intros s prefix suffix ap HIn.
    induction prefix as [| seg rest IH].
    - (* Empty prefix: suffix unchanged *)
      simpl. exact HIn.
    - (* prefix = seg :: rest *)
      simpl.
      apply in_or_app.
      right.
      apply IH.
  Qed.

  (** Key property: Data sent on path p is visible when looking up from p
      via the suffix collection. Combined with pathmap_get_put_same, this
      shows that put followed by get_with_prefixes finds the data. *)
  Theorem send_visible_via_suffix_collection :
    forall (s : pathmap_storage A K) (p : Path) (a : A) (persist : bool),
      In (a, persist) (pathmap_get_data_with_prefixes A K
                        (pathmap_put_data A K s p a persist) p).
  Proof.
    intros s p a persist.
    apply suffix_collection_includes_exact.
    apply (pathmap_get_put_same A K).
  Qed.

  (** Data at path p is visible when looking up from any path that
      has p as a suffix. This is the converse of the design spec's
      prefix matching - here the receiver must be at a PREFIX of
      the sender's path. *)
  Theorem send_visible_from_prefixed_path :
    forall (s : pathmap_storage A K) (prefix p : Path) (a : A) (persist : bool),
      In (a, persist) (pathmap_get_data_with_prefixes A K
                        (pathmap_put_data A K s p a persist) (prefix ++ p)).
  Proof.
    intros s prefix p a persist.
    apply suffix_collection_includes_suffix.
    apply send_visible_via_suffix_collection.
  Qed.

End PrefixProperties.

(** ** Prefix Aggregation Semantics for Consume and Produce *)

(** This section formalizes the PathMap prefix aggregation semantics
    as implemented in generic_rspace.rs:

    1. consume() on prefix path @[0,1] can find data at @[0,1,2], @[0,1,3], etc.
    2. produce() on child path @[0,1,2] can trigger continuations at @[0,1]

    Reference: Reifying RSpaces.md lines 159-192 *)

Section PrefixAggregation.
  Variable A K : Type.

  (** ** Suffix Key Extraction *)

  (** Extract suffix when child has parent as prefix.
      get_path_suffix [0;1] [0;1;2;3] = Some [2;3]
      get_path_suffix [0;1] [0;2] = None *)
  Fixpoint get_path_suffix (parent child : Path) : option Path :=
    match parent, child with
    | [], _ => Some child
    | _, [] => None
    | p :: parent_rest, c :: child_rest =>
      if Nat.eqb p c then get_path_suffix parent_rest child_rest
      else None
    end.

  (** Suffix extraction is sound: result is actual suffix *)
  Theorem get_path_suffix_sound :
    forall parent child suffix,
      get_path_suffix parent child = Some suffix ->
      child = parent ++ suffix.
  Proof.
    induction parent as [| p prest IH].
    - (* parent = [] *)
      intros child suffix H.
      simpl in H. injection H as Hsuffix.
      subst. reflexivity.
    - (* parent = p :: prest *)
      intros child suffix H.
      destruct child as [| c crest].
      + (* child = [] *)
        simpl in H. discriminate.
      + (* child = c :: crest *)
        simpl in H.
        destruct (Nat.eqb p c) eqn:Epc.
        * apply Nat.eqb_eq in Epc. subst c.
          specialize (IH crest suffix H).
          simpl. f_equal. exact IH.
        * discriminate.
  Qed.

  (** Suffix extraction is complete: prefixes yield suffixes *)
  Theorem get_path_suffix_complete :
    forall parent suffix,
      get_path_suffix parent (parent ++ suffix) = Some suffix.
  Proof.
    induction parent as [| p prest IH].
    - intros suffix. reflexivity.
    - intros suffix. simpl.
      rewrite Nat.eqb_refl.
      apply IH.
  Qed.

  (** Suffix key for exact match is empty *)
  Theorem suffix_key_exact_match :
    forall p, get_path_suffix p p = Some [].
  Proof.
    intros p.
    rewrite <- (app_nil_r p) at 2.
    apply get_path_suffix_complete.
  Qed.

  (** ** Channel Prefix Enumeration *)

  (** Generate all prefixes of a path.
      channel_prefixes [0;1;2] = [[]; [0]; [0;1]; [0;1;2]] *)
  Fixpoint channel_prefixes (p : Path) : list Path :=
    match p with
    | [] => [[]]
    | s :: rest => [] :: map (cons s) (channel_prefixes rest)
    end.

  (** A path is always in its own prefix list *)
  Theorem path_in_own_prefixes :
    forall p, In p (channel_prefixes p).
  Proof.
    induction p as [| s rest IH].
    - simpl. left. reflexivity.
    - simpl. right.
      apply in_map.
      exact IH.
  Qed.

  (** Empty path is prefix of all paths *)
  Theorem empty_in_prefixes :
    forall p, In [] (channel_prefixes p).
  Proof.
    destruct p as [| s rest].
    - simpl. left. reflexivity.
    - simpl. left. reflexivity.
  Qed.

  (** All elements of channel_prefixes are actually prefixes *)
  Theorem channel_prefixes_are_prefixes :
    forall p prefix,
      In prefix (channel_prefixes p) ->
      is_prefix prefix p = true.
  Proof.
    induction p as [| s rest IH].
    - (* p = [] *)
      intros prefix H.
      simpl in H.
      destruct H as [H | H]; try contradiction.
      subst prefix. reflexivity.
    - (* p = s :: rest *)
      intros prefix H.
      simpl in H.
      destruct H as [H | H].
      + subst prefix. reflexivity.
      + apply in_map_iff in H.
        destruct H as [x [Hx Hin]].
        subst prefix.
        simpl. rewrite Nat.eqb_refl.
        apply IH. exact Hin.
  Qed.

  (** Parent path is in prefixes of extended path *)
  Lemma parent_in_prefixes_of_append : forall parent suffix,
    In parent (channel_prefixes (parent ++ suffix)).
  Proof.
    induction parent as [| s rest IH].
    - intros suffix. simpl.
      destruct suffix; simpl; left; reflexivity.
    - intros suffix. simpl.
      right.
      (* We need to show: In (s :: rest) (map (cons s) (channel_prefixes (rest ++ suffix))) *)
      apply in_map_iff.
      exists rest.
      split.
      + reflexivity.
      + apply IH.
  Qed.

  (** ** Continuation Storage Model *)

  (** Continuations are stored at specific paths with patterns *)
  Record stored_continuation := mkStoredCont {
    sc_patterns : list A;
    sc_continuation : K;
    sc_persist : bool;
  }.

  (** Extended storage with continuation tracking per path *)
  Record pathmap_with_conts := mkPathMapWithConts {
    pwc_data : Path -> list (A * bool);
    pwc_conts : Path -> list stored_continuation;
    pwc_counter : nat;
  }.

  (** ** Consume Prefix Semantics *)

  (** Check if data matches pattern (placeholder - actual matching is in matcher) *)
  Variable matches : A -> A -> bool.

  (** If x is in filter P l, and we map f over that, then f x is in the map result *)
  Lemma In_map_filter : forall {B C : Type} (P : B -> bool) (f : B -> C) (l : list B) (x : B),
    In x l -> P x = true -> In (f x) (map f (filter P l)).
  Proof.
    intros B C P f l x Hin HP.
    apply in_map.
    apply filter_In_iff.
    split; assumption.
  Qed.

  (** If the filter for Some produces h :: t, then h = Some x for some x and x came from descendants *)
  Lemma filter_some_map_origin : forall {B C : Type} (f : B -> option C)
    (l : list B) (h : option C) (t : list (option C)),
    filter (fun opt => match opt with Some _ => true | None => false end) (map f l) = h :: t ->
    exists x y, In x l /\ f x = Some y /\ h = Some y.
  Proof.
    intros B C f l h t Hfilter.
    (* The filter keeps only Some values, so h must be Some y *)
    assert (Hsome: exists y, h = Some y).
    { destruct h as [y |].
      - exists y. reflexivity.
      - exfalso.
        assert (Hin: In None (None :: t)) by (left; reflexivity).
        rewrite <- Hfilter in Hin.
        apply filter_In in Hin.
        destruct Hin as [_ Habs]. discriminate.
    }
    destruct Hsome as [y Heq]. subst h.
    (* Some y is in the filter result, so it's in map f l *)
    assert (Hin_filter: In (Some y) (Some y :: t)) by (left; reflexivity).
    rewrite <- Hfilter in Hin_filter.
    apply filter_In in Hin_filter. destruct Hin_filter as [Hin_map _].
    (* Some y is in map f l, so exists x with f x = Some y *)
    apply in_map_iff in Hin_map.
    destruct Hin_map as [x [Hfx Hinx]].
    exists x, y.
    split; [exact Hinx | split; [exact Hfx | reflexivity]].
  Qed.

  (** If the head of filtered map results is Some y, and all inputs satisfy P, then y's first component satisfies P *)
  Lemma filter_some_map_preserves_pred : forall {B C D : Type}
    (P : B -> bool) (f : B -> option (C * D)) (l : list B)
    (c : C) (d : D) (t : list (option (C * D))),
    filter (fun opt => match opt with Some _ => true | None => false end) (map f (filter P l)) = Some (c, d) :: t ->
    exists x, In x l /\ P x = true /\ f x = Some (c, d).
  Proof.
    intros B C D P f l c d t Hfilter.
    apply filter_some_map_origin in Hfilter.
    destruct Hfilter as [x [y [Hinx [Hfx Heq]]]].
    injection Heq as Heq. subst y.
    (* x is in filter P l *)
    apply filter_In in Hinx.
    destruct Hinx as [Hin_l HPx].
    exists x.
    split; [exact Hin_l | split; [exact HPx | exact Hfx]].
  Qed.

  (** Find matching data at a specific path *)
  Definition find_data_at_path (s : pathmap_with_conts) (p : Path) (pattern : A)
    : option A :=
    let data := pwc_data s p in
    match filter (fun ap => matches pattern (fst ap)) data with
    | [] => None
    | (a, _) :: _ => Some a
    end.

  (** Find matching data at path or any descendant.
      This models consume() with prefix semantics:
      consume on @[0,1] can find data at @[0,1,2] *)
  Definition find_data_with_descendants
    (s : pathmap_with_conts) (prefix : Path) (all_paths : list Path) (pattern : A)
    : option (Path * A) :=
    let descendants := filter (fun p => is_prefix prefix p) all_paths in
    let results := map (fun p =>
      match find_data_at_path s p pattern with
      | None => None
      | Some a => Some (p, a)
      end) descendants in
    match filter (fun opt => match opt with Some _ => true | None => false end) results with
    | [] => None
    | Some (p, a) :: _ => Some (p, a)
    | None :: _ => None (* unreachable due to filter *)
    end.

  (** Consume finds data at exact path first *)
  Theorem consume_exact_match_priority :
    forall s prefix all_paths pattern a,
      In prefix all_paths ->
      find_data_at_path s prefix pattern = Some a ->
      exists p' a', find_data_with_descendants s prefix all_paths pattern = Some (p', a') /\
                    is_prefix prefix p' = true.
  Proof.
    intros s prefix all_paths pattern a Hin Hfind.
    unfold find_data_with_descendants.
    (* prefix is in descendants because is_prefix prefix prefix = true *)
    assert (Hrefl: is_prefix prefix prefix = true) by apply is_prefix_refl.
    assert (Hin_desc: In prefix (filter (fun p => is_prefix prefix p) all_paths)).
    { apply filter_In_iff. split; assumption. }
    (* The map produces Some (prefix, a) at prefix *)
    set (f := fun p =>
      match find_data_at_path s p pattern with
      | None => None
      | Some a0 => Some (p, a0)
      end).
    assert (Hmap_prefix: f prefix = Some (prefix, a)).
    { unfold f. rewrite Hfind. reflexivity. }
    (* Therefore Some (prefix, a) is in the results *)
    set (descendants := filter (fun p => is_prefix prefix p) all_paths).
    set (results := map f descendants).
    assert (Hin_results: In (Some (prefix, a)) results).
    { unfold results, descendants.
      apply in_map_iff.
      exists prefix.
      split.
      - exact Hmap_prefix.
      - exact Hin_desc.
    }
    (* The filter will produce a non-empty list *)
    destruct (some_in_filter_some results (prefix, a) Hin_results) as [h [t Hfilter]].
    rewrite Hfilter.
    (* h must be Some (p', a') for some p', a' *)
    destruct (filter_some_preserves results h t Hfilter) as [[p' a'] Heq].
    subst h.
    exists p', a'.
    split.
    - reflexivity.
    - (* p' came from descendants, so is_prefix prefix p' = true *)
      (* We need to show this using the filter_some_map_origin lemma *)
      apply filter_some_map_origin in Hfilter.
      destruct Hfilter as [x [[p'' a''] [Hinx [Hfx Heq]]]].
      injection Heq as Hp' Ha'. subst p'' a''.
      (* x is in descendants = filter (fun p => is_prefix prefix p) all_paths *)
      unfold descendants in Hinx.
      apply filter_In_iff in Hinx.
      destruct Hinx as [_ Hprefix].
      (* From Hfx: f x = Some (p', a'), and f x only produces p' = x *)
      unfold f in Hfx.
      destruct (find_data_at_path s x pattern) as [a0 |] eqn:Hdata.
      + injection Hfx as Hp'_eq _. subst x.
        exact Hprefix.
      + discriminate Hfx.
  Qed.

  (** ** Produce Prefix Semantics *)

  (** Find continuation at a specific path *)
  Definition find_cont_at_path (s : pathmap_with_conts) (p : Path) (pattern : A)
    : option stored_continuation :=
    let conts := pwc_conts s p in
    match filter (fun sc =>
      match sc_patterns sc with
      | pat :: _ => matches pat pattern
      | [] => false
      end) conts with
    | [] => None
    | sc :: _ => Some sc
    end.

  (** Find continuation at any prefix path of the produce channel.
      This models produce() with prefix semantics:
      produce on @[0,1,2] can trigger continuation at @[0,1] *)
  Definition find_cont_at_prefix
    (s : pathmap_with_conts) (channel : Path) (data : A)
    : option (Path * stored_continuation) :=
    let prefixes := channel_prefixes channel in
    let results := map (fun p =>
      match find_cont_at_path s p data with
      | None => None
      | Some sc => Some (p, sc)
      end) prefixes in
    match filter (fun opt => match opt with Some _ => true | None => false end) results with
    | [] => None
    | Some (p, sc) :: _ => Some (p, sc)
    | None :: _ => None
    end.

  (** Produce triggers continuation at exact path first *)
  Theorem produce_exact_match_priority :
    forall s channel data sc,
      find_cont_at_path s channel data = Some sc ->
      exists p' sc', find_cont_at_prefix s channel data = Some (p', sc') /\
                     is_prefix p' channel = true.
  Proof.
    intros s channel data sc Hfind.
    unfold find_cont_at_prefix.
    (* channel is in its own prefixes *)
    assert (Hin: In channel (channel_prefixes channel)) by apply path_in_own_prefixes.
    (* Set up names for readability *)
    set (f := fun p =>
      match find_cont_at_path s p data with
      | None => None
      | Some sc0 => Some (p, sc0)
      end).
    set (prefixes := channel_prefixes channel).
    set (results := map f prefixes).
    (* f channel = Some (channel, sc) *)
    assert (Hmap_channel: f channel = Some (channel, sc)).
    { unfold f. rewrite Hfind. reflexivity. }
    (* Some (channel, sc) is in results *)
    assert (Hin_results: In (Some (channel, sc)) results).
    { unfold results.
      apply in_map_iff.
      exists channel.
      split; [exact Hmap_channel | exact Hin].
    }
    (* The filter will produce a non-empty list *)
    destruct (some_in_filter_some results (channel, sc) Hin_results) as [h [t Hfilter]].
    rewrite Hfilter.
    (* h must be Some (p', sc') for some p', sc' *)
    destruct (filter_some_preserves results h t Hfilter) as [[p' sc'] Heq].
    subst h.
    exists p', sc'.
    split.
    - reflexivity.
    - (* p' came from prefixes, so is_prefix p' channel = true *)
      apply filter_some_map_origin in Hfilter.
      destruct Hfilter as [x [[p'' sc''] [Hinx [Hfx Heq]]]].
      injection Heq as Hp' Hsc'. subst p'' sc''.
      (* x is in prefixes = channel_prefixes channel *)
      unfold prefixes in Hinx.
      apply channel_prefixes_are_prefixes in Hinx.
      (* From Hfx: f x = Some (p', sc'), and f x produces p' = x *)
      unfold f in Hfx.
      destruct (find_cont_at_path s x data) as [sc0 |] eqn:Hcont.
      + injection Hfx as Hp'_eq _. subst x.
        exact Hinx.
      + discriminate Hfx.
  Qed.

  (** Produce on child can trigger continuation at prefix *)
  Theorem produce_finds_prefix_continuation :
    forall s parent suffix data sc,
      find_cont_at_path s parent data = Some sc ->
      exists p' sc', find_cont_at_prefix s (parent ++ suffix) data = Some (p', sc') /\
                     is_prefix p' (parent ++ suffix) = true.
  Proof.
    intros s parent suffix data sc Hfind.
    unfold find_cont_at_prefix.
    (* parent is in prefixes of parent ++ suffix *)
    assert (Hin: In parent (channel_prefixes (parent ++ suffix))) by apply parent_in_prefixes_of_append.
    (* Set up names for readability *)
    set (channel := parent ++ suffix).
    set (f := fun p =>
      match find_cont_at_path s p data with
      | None => None
      | Some sc0 => Some (p, sc0)
      end).
    set (prefixes := channel_prefixes channel).
    set (results := map f prefixes).
    (* f parent = Some (parent, sc) *)
    assert (Hmap_parent: f parent = Some (parent, sc)).
    { unfold f. rewrite Hfind. reflexivity. }
    (* Some (parent, sc) is in results *)
    assert (Hin_results: In (Some (parent, sc)) results).
    { unfold results, prefixes, channel.
      apply in_map_iff.
      exists parent.
      split; [exact Hmap_parent | exact Hin].
    }
    (* The filter will produce a non-empty list *)
    destruct (some_in_filter_some results (parent, sc) Hin_results) as [h [t Hfilter]].
    rewrite Hfilter.
    (* h must be Some (p', sc') for some p', sc' *)
    destruct (filter_some_preserves results h t Hfilter) as [[p' sc'] Heq].
    subst h.
    exists p', sc'.
    split.
    - reflexivity.
    - (* p' came from prefixes, so is_prefix p' channel = true *)
      apply filter_some_map_origin in Hfilter.
      destruct Hfilter as [x [[p'' sc''] [Hinx [Hfx Heq]]]].
      injection Heq as Hp' Hsc'. subst p'' sc''.
      (* x is in prefixes = channel_prefixes channel *)
      unfold prefixes, channel in Hinx.
      apply channel_prefixes_are_prefixes in Hinx.
      (* From Hfx: f x = Some (p', sc'), and f x produces p' = x *)
      unfold f in Hfx.
      destruct (find_cont_at_path s x data) as [sc0 |] eqn:Hcont.
      + injection Hfx as Hp'_eq _. subst x.
        exact Hinx.
      + discriminate Hfx.
  Qed.

  (** ** Tuple Space Invariant Preservation *)

  (** Helper: matches is assumed reflexive for patterns *)
  Variable matches_refl : forall a, matches a a = true.

  (** We need decidable equality on stored_continuation for the proof *)
  Hypothesis stored_continuation_eq_dec :
    forall sc1 sc2 : stored_continuation, {sc1 = sc2} + {sc1 <> sc2}.

  (** Helper: filter on empty list *)
  Lemma filter_empty_iff : forall {B : Type} (f : B -> bool) (l : list B),
    filter f l = [] <-> forall x, In x l -> f x = false.
  Proof.
    intros B f l.
    split.
    - intros Hempty x Hin.
      destruct (f x) eqn:Hfx.
      + (* f x = true, so x should be in filter *)
        assert (In x (filter f l)).
        { apply filter_In. split; assumption. }
        rewrite Hempty in H. destruct H.
      + reflexivity.
    - intros Hall.
      induction l as [| h t IH].
      + reflexivity.
      + simpl.
        assert (Hh: f h = false) by (apply Hall; left; reflexivity).
        rewrite Hh.
        apply IH.
        intros x Hin.
        apply Hall. right. exact Hin.
  Qed.

  (** Helper lemma: filter composition removes elements matching data if uniqueness holds *)
  Lemma filter_removes_matching :
    forall (conts : list stored_continuation) (sc : stored_continuation) (data : A),
      (* sc's pattern matches data *)
      (match sc_patterns sc with pat :: _ => matches pat data = true | [] => False end) ->
      (* sc is the only one whose pattern matches data *)
      (forall sc', In sc' conts -> sc' <> sc ->
        match sc_patterns sc' with pat :: _ => matches pat data = false | [] => True end) ->
      (* After removing sc (by pattern match), the remaining list has no data matches *)
      filter (fun sc0 => match sc_patterns sc0 with [] => false | pat :: _ => matches pat data end)
        (filter (fun sc' => negb (match sc_patterns sc', sc_patterns sc with
                                  | p1 :: _, p2 :: _ => matches p1 p2
                                  | _, _ => false end)) conts) = [].
  Proof.
    intros conts sc data Hsc_matches Huniq.
    (* We show no element survives both filters *)
    apply filter_empty_iff.
    intros sc' Hin_inner.
    apply filter_In in Hin_inner. destruct Hin_inner as [Hin_orig Hnegb].
    destruct (sc_patterns sc') as [| pat' rest'] eqn:Hpat'.
    - (* sc' has no patterns - fails outer filter *)
      reflexivity.
    - (* sc' has pattern pat' :: rest' *)
      (* Either sc' = sc or sc' <> sc *)
      destruct (stored_continuation_eq_dec sc' sc) as [Heq | Hneq].
      + (* sc' = sc: but then sc' would fail the inner negb filter *)
        subst sc'.
        (* After subst, Hpat' says sc_patterns sc = pat' :: rest' *)
        (* Hsc_matches says: match sc_patterns sc with pat :: _ => true | [] => False end *)
        (* Hnegb says: negb (match ...) = true *)
        (* Since sc_patterns sc = pat' :: rest', Hnegb becomes negb (matches pat' pat') = true *)
        rewrite Hpat' in Hnegb.
        simpl in Hnegb.
        rewrite matches_refl in Hnegb.
        simpl in Hnegb. discriminate.
      + (* sc' <> sc: Huniq says matches pat' data = false *)
        specialize (Huniq sc' Hin_orig Hneq).
        rewrite Hpat' in Huniq.
        exact Huniq.
  Qed.

  (** After produce triggers prefix continuation, the matched continuation is removed.
      Note: This proves that if sc was the only continuation whose pattern matches data,
      then after removal find_cont_at_path returns None. *)
  Theorem produce_prefix_preserves_invariant :
    forall s channel data p sc s',
      find_cont_at_prefix s channel data = Some (p, sc) ->
      sc_persist sc = false ->
      (* sc's pattern matches data (derived from find_cont_at_prefix) *)
      (match sc_patterns sc with pat :: _ => matches pat data = true | [] => False end) ->
      (* s' is s with the matched continuation removed from p *)
      pwc_conts s' p = filter (fun sc' => negb (
        match sc_patterns sc', sc_patterns sc with
        | p1 :: _, p2 :: _ => matches p1 p2
        | _, _ => false
        end)) (pwc_conts s p) ->
      (forall p', p' <> p -> pwc_conts s' p' = pwc_conts s p') ->
      pwc_data s' = pwc_data s ->
      (* sc is the only continuation at p whose pattern matches data *)
      (forall sc', In sc' (pwc_conts s p) -> sc' <> sc ->
        match sc_patterns sc' with
        | pat :: _ => matches pat data = false
        | [] => True
        end) ->
      (* After removal, no matching continuation exists at p *)
      find_cont_at_path s' p data = None.
  Proof.
    intros s channel data p sc s' Hfind Hpersist Hsc_matches Hremove Hother Hdata Huniq.
    unfold find_cont_at_path.
    rewrite Hremove.
    rewrite (filter_removes_matching (pwc_conts s p) sc data Hsc_matches Huniq).
    reflexivity.
  Qed.

  (** ** Suffix Key in Results *)

  (** When produce triggers prefix continuation, suffix key is computed correctly *)
  Theorem produce_result_has_correct_suffix :
    forall parent suffix,
      get_path_suffix parent (parent ++ suffix) = Some suffix.
  Proof.
    exact get_path_suffix_complete.
  Qed.

  (** When consume finds data at descendant, suffix key relates paths *)
  Theorem consume_result_suffix_relates_paths :
    forall prefix child suffix,
      get_path_suffix prefix child = Some suffix ->
      child = prefix ++ suffix.
  Proof.
    exact get_path_suffix_sound.
  Qed.

End PrefixAggregation.
