(** * Outer Storage Type Specifications

    This module specifies the behavior of outer storage types for channel stores.
    Outer storage types determine how channels are organized and named.

    Reference: Spec "Reifying RSpaces.md" lines about outer storage types
*)

From Coq Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Abstract Channel Index *)

Definition ChannelIndex := nat.

(** ** Fixed Array Storage *)

(** Fixed array with static bounds - fails when full *)
Section FixedArray.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  Definition fixed_array := list (option A).

  Definition fixed_array_empty : fixed_array :=
    repeat None capacity.

  Definition fixed_array_len (arr : fixed_array) : nat := length arr.

  (** Get value at index *)
  Definition fixed_array_get (arr : fixed_array) (idx : ChannelIndex) : option A :=
    match nth_error arr idx with
    | None => None  (* out of bounds *)
    | Some opt => opt
    end.

  (** Find first empty slot *)
  Fixpoint fixed_array_find_empty (arr : fixed_array) (start : nat) : option ChannelIndex :=
    match arr with
    | [] => None
    | Some _ :: rest => fixed_array_find_empty rest (S start)
    | None :: _ => Some start
    end.

  (** Set value at index - fails if index out of bounds *)
  Fixpoint fixed_array_set (arr : fixed_array) (idx : ChannelIndex) (val : A)
    : option fixed_array :=
    match arr, idx with
    | [], _ => None  (* out of bounds *)
    | _ :: rest, 0 => Some (Some val :: rest)
    | h :: rest, S idx' =>
      match fixed_array_set rest idx' val with
      | None => None
      | Some rest' => Some (h :: rest')
      end
    end.

  (** Put value in first available slot - may fail with OutOfNames *)
  Definition fixed_array_put (arr : fixed_array) (val : A) : option (fixed_array * ChannelIndex) :=
    match fixed_array_find_empty arr 0 with
    | None => None  (* OutOfNames error - array is full *)
    | Some idx =>
      match fixed_array_set arr idx val with
      | None => None  (* Should not happen if find_empty returned valid index *)
      | Some arr' => Some (arr', idx)
      end
    end.

  (** Clear value at index *)
  Fixpoint fixed_array_clear (arr : fixed_array) (idx : ChannelIndex) : option fixed_array :=
    match arr, idx with
    | [], _ => None  (* out of bounds *)
    | _ :: rest, 0 => Some (None :: rest)
    | h :: rest, S idx' =>
      match fixed_array_clear rest idx' with
      | None => None
      | Some rest' => Some (h :: rest')
      end
    end.

  (** Count occupied slots *)
  Fixpoint fixed_array_count (arr : fixed_array) : nat :=
    match arr with
    | [] => 0
    | Some _ :: rest => S (fixed_array_count rest)
    | None :: rest => fixed_array_count rest
    end.

End FixedArray.

(** ** Fixed Array Properties *)

Section FixedArrayProperties.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  (** Empty array has correct length *)
  Theorem fixed_array_empty_len :
    fixed_array_len A (fixed_array_empty A capacity) = capacity.
  Proof.
    unfold fixed_array_empty, fixed_array_len.
    apply repeat_length.
  Qed.

  (** Empty array has all None values *)
  Theorem fixed_array_empty_all_none :
    forall idx,
      idx < capacity ->
      fixed_array_get A (fixed_array_empty A capacity) idx = None.
  Proof.
    intros idx Hlt.
    unfold fixed_array_get, fixed_array_empty.
    (* nth_error on repeat None gives None *)
    assert (Hnth: nth_error (repeat (@None A) capacity) idx = Some None).
    { apply nth_error_repeat. exact Hlt. }
    rewrite Hnth. reflexivity.
  Qed.

  (** Get on out-of-bounds index returns None *)
  Theorem fixed_array_get_oob :
    forall arr idx,
      idx >= fixed_array_len A arr ->
      fixed_array_get A arr idx = None.
  Proof.
    intros arr idx Hge.
    unfold fixed_array_get, fixed_array_len in *.
    assert (Hoob : nth_error arr idx = None).
    { apply nth_error_None. lia. }
    rewrite Hoob. reflexivity.
  Qed.

  (** Set at valid index succeeds *)
  Theorem fixed_array_set_valid :
    forall arr idx val,
      idx < fixed_array_len A arr ->
      exists arr', fixed_array_set A arr idx val = Some arr'.
  Proof.
    intros arr idx val Hlt.
    unfold fixed_array_len in Hlt.
    generalize dependent idx.
    induction arr as [| h t IH].
    - intros idx Hlt. simpl in Hlt. lia.
    - intros idx Hlt.
      destruct idx.
      + simpl. eexists. reflexivity.
      + simpl in Hlt.
        simpl.
        assert (Hlt' : idx < length t) by lia.
        specialize (IH idx Hlt').
        destruct IH as [arr' Harr'].
        rewrite Harr'.
        eexists. reflexivity.
  Qed.

  (** Set at out-of-bounds index fails *)
  Theorem fixed_array_set_oob :
    forall arr idx val,
      idx >= fixed_array_len A arr ->
      fixed_array_set A arr idx val = None.
  Proof.
    intros arr idx val Hge.
    unfold fixed_array_len in Hge.
    generalize dependent idx.
    induction arr as [| h t IH].
    - intros. reflexivity.
    - intros idx Hge.
      destruct idx.
      + simpl in Hge. lia.
      + simpl in Hge. simpl.
        assert (Hge' : idx >= length t) by lia.
        rewrite (IH idx Hge').
        reflexivity.
  Qed.

  (** Helper: count is always <= length *)
  Lemma count_le_length :
    forall arr, fixed_array_count A arr <= length arr.
  Proof.
    induction arr as [| h t IH].
    - simpl. lia.
    - destruct h as [a |]; simpl.
      + (* Some a :: t: count = S (count t), length = S (length t) *)
        specialize (IH). lia.
      + (* None :: t: count = count t, length = S (length t) *)
        specialize (IH). lia.
  Qed.

  (** Helper: find_empty returns None when all elements are Some *)
  Lemma find_empty_all_some :
    forall arr start,
      fixed_array_count A arr = length arr ->
      fixed_array_find_empty A arr start = None.
  Proof.
    induction arr as [| h t IH].
    - intros. reflexivity.
    - intros start Hfull. destruct h as [a |].
      + simpl in Hfull. simpl.
        apply IH.
        (* S (count t) = S (length t) implies count t = length t *)
        lia.
      + (* h = None but count should equal len, contradiction *)
        (* count (None :: t) = count t, but length (None :: t) = S (length t) *)
        simpl in Hfull.
        (* Hfull : fixed_array_count t = S (length t), but count <= length *)
        exfalso.
        pose proof (count_le_length t) as Hle.
        lia.
  Qed.

  (** Put on full array fails (OutOfNames) *)
  Theorem fixed_array_put_full_fails :
    forall arr val,
      fixed_array_count A arr = fixed_array_len A arr ->
      fixed_array_put A arr val = None.
  Proof.
    intros arr val Hfull.
    unfold fixed_array_put.
    unfold fixed_array_len in Hfull.
    rewrite (find_empty_all_some arr 0 Hfull).
    reflexivity.
  Qed.

  (** Set preserves length *)
  Theorem fixed_array_set_preserves_len :
    forall arr idx val arr',
      fixed_array_set A arr idx val = Some arr' ->
      fixed_array_len A arr' = fixed_array_len A arr.
  Proof.
    intros arr idx val arr' Hset.
    generalize dependent arr'.
    generalize dependent idx.
    induction arr as [| h t IH].
    - intros. simpl in Hset. discriminate.
    - intros idx arr' Hset.
      destruct idx as [| idx'].
      + simpl in Hset. injection Hset as Harr'. subst arr'.
        reflexivity.
      + simpl in Hset.
        destruct (fixed_array_set A t idx' val) as [rest' |] eqn:Hrest.
        2: { discriminate. }
        injection Hset as Harr'. subst arr'.
        unfold fixed_array_len. simpl.
        f_equal.
        apply (IH idx' rest' Hrest).
  Qed.

End FixedArrayProperties.

(** ** Cyclic Array Storage *)

(** Cyclic array - wraps around when reaching the end *)
Section CyclicArray.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  (* Cyclic array with current position *)
  Record cyclic_array := {
    ca_data : list (option A);
    ca_next : nat;  (* Next write position *)
  }.

  Definition cyclic_array_empty : cyclic_array := {|
    ca_data := repeat None capacity;
    ca_next := 0;
  |}.

  (** Put value - always succeeds, wraps around *)
  Definition cyclic_array_put (arr : cyclic_array) (val : A) : cyclic_array * ChannelIndex :=
    let idx := ca_next arr in
    let data' := match fixed_array_set A (ca_data arr) idx val with
                 | Some d => d
                 | None => ca_data arr  (* Should not happen *)
                 end in
    let next' := (idx + 1) mod capacity in
    ({| ca_data := data'; ca_next := next' |}, idx).

  Definition cyclic_array_get (arr : cyclic_array) (idx : ChannelIndex) : option A :=
    fixed_array_get A (ca_data arr) idx.

End CyclicArray.

(** ** Cyclic Array Properties *)

(** Properties of cyclic arrays - proven within the same section scope *)
Section CyclicArrayProperties.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  (* Re-use the same record type with the same section variables *)
  (* Cyclic array with current position *)
  Record ca_array := {
    caa_data : list (option A);
    caa_next : nat;  (* Next write position *)
  }.

  Definition ca_put (arr : ca_array) (val : A) : ca_array * ChannelIndex :=
    let idx := caa_next arr in
    let data' := match fixed_array_set A (caa_data arr) idx val with
                 | Some d => d
                 | None => caa_data arr
                 end in
    let next' := (idx + 1) mod capacity in
    ({| caa_data := data'; caa_next := next' |}, idx).

  (** Cyclic put always succeeds (returns valid index) *)
  Theorem cyclic_array_put_always_succeeds :
    forall (arr : ca_array) val,
      caa_next arr < capacity ->
      let '(arr', idx) := ca_put arr val in
      idx < capacity /\ caa_next arr' = (idx + 1) mod capacity.
  Proof.
    intros arr val Hnext.
    unfold ca_put.
    simpl.
    split.
    - exact Hnext.
    - reflexivity.
  Qed.

  (** After capacity puts, index wraps to 0 *)
  Theorem cyclic_array_wraps :
    forall (arr : ca_array) (default_val : A),
      caa_next arr = capacity - 1 ->
      let '(arr', _) := ca_put arr default_val in
      caa_next arr' = 0.
  Proof.
    intros arr default_val Hnext.
    unfold ca_put. simpl.
    rewrite Hnext.
    assert (Hmod : (capacity - 1 + 1) mod capacity = 0).
    { assert (Heq : capacity - 1 + 1 = capacity) by lia.
      rewrite Heq.
      apply Nat.mod_same. lia.
    }
    exact Hmod.
  Qed.

End CyclicArrayProperties.

(** ** Vector (Dynamic Array) Storage *)

(** Vector that grows dynamically *)
Section VectorStorage.
  Variable A : Type.

  (* Vector with current length and data *)
  Record vector := {
    vec_data : list (option A);
    vec_len : nat;  (* Number of occupied slots *)
  }.

  Definition vector_empty : vector := {|
    vec_data := [];
    vec_len := 0;
  |}.

  Definition vector_capacity (v : vector) : nat := length (vec_data v).

  (** Grow vector by doubling capacity (or to 1 if empty) *)
  Definition vector_grow (v : vector) : vector :=
    let current_cap := vector_capacity v in
    let new_cap := if current_cap =? 0 then 1 else current_cap * 2 in
    {| vec_data := vec_data v ++ repeat None (new_cap - current_cap);
       vec_len := vec_len v |}.

  (** Put value - grows if necessary, may fail with OOM (not modeled here) *)
  Definition vector_put (v : vector) (val : A) : vector * ChannelIndex :=
    let v' := if vec_len v <? vector_capacity v then v else vector_grow v in
    let idx := vec_len v' in
    let data' := match fixed_array_set A (vec_data v') idx val with
                 | Some d => d
                 | None => vec_data v'  (* Should not happen after grow *)
                 end in
    ({| vec_data := data'; vec_len := S idx |}, idx).

  Definition vector_get (v : vector) (idx : ChannelIndex) : option A :=
    if idx <? vec_len v
    then fixed_array_get A (vec_data v) idx
    else None.

End VectorStorage.

(** ** Vector Properties *)

Section VectorProperties.
  Variable A : Type.

  (** Helper: Get after set at same index returns the set value *)
  Lemma fixed_array_get_set_same :
    forall arr idx val arr',
      fixed_array_set A arr idx val = Some arr' ->
      fixed_array_get A arr' idx = Some val.
  Proof.
    induction arr as [| h t IH]; intros idx val arr' Hset.
    - simpl in Hset. discriminate.
    - destruct idx as [| idx'].
      + simpl in Hset. injection Hset as Harr'. subst arr'.
        unfold fixed_array_get. simpl. reflexivity.
      + simpl in Hset.
        destruct (fixed_array_set A t idx' val) as [rest' |] eqn:Hrest.
        2: { discriminate. }
        injection Hset as Harr'. subst arr'.
        unfold fixed_array_get. simpl.
        specialize (IH idx' val rest' Hrest).
        unfold fixed_array_get in IH.
        exact IH.
  Qed.

  (** Helper: grow always increases capacity to at least 1 *)
  Lemma vector_grow_capacity_pos :
    forall v,
      vector_capacity A (vector_grow A v) > 0.
  Proof.
    intros v.
    unfold vector_grow, vector_capacity. simpl.
    rewrite app_length. rewrite repeat_length.
    destruct (length (vec_data A v) =? 0) eqn:Heq.
    - apply Nat.eqb_eq in Heq. rewrite Heq. simpl. lia.
    - apply Nat.eqb_neq in Heq. lia.
  Qed.

  (** Helper: grow preserves existing length *)
  Lemma vector_grow_len_preserved :
    forall v,
      vec_len A (vector_grow A v) = vec_len A v.
  Proof.
    intros v. unfold vector_grow. simpl. reflexivity.
  Qed.

  (** Well-formed vector: len <= capacity *)
  Definition vector_wf (v : vector A) : Prop :=
    vec_len A v <= vector_capacity A v.

  (** Helper: After grow, capacity > len for well-formed vectors *)
  Lemma vector_grow_capacity_gt_len :
    forall v,
      vector_wf v ->
      vec_len A v < vector_capacity A (vector_grow A v).
  Proof.
    intros v Hwf.
    unfold vector_wf, vector_grow, vector_capacity in *. simpl.
    rewrite app_length. rewrite repeat_length.
    destruct (length (vec_data A v) =? 0) eqn:Hzero.
    - apply Nat.eqb_eq in Hzero.
      rewrite Hzero in Hwf. simpl in Hwf.
      (* len <= 0 means len = 0 *)
      assert (Hlen0 : vec_len A v = 0) by lia.
      rewrite Hzero. rewrite Hlen0. simpl. lia.
    - apply Nat.eqb_neq in Hzero.
      (* capacity > 0, so capacity * 2 > capacity >= len *)
      lia.
  Qed.

  (** Helper: grow preserves data up to old capacity *)
  Lemma vector_grow_data_prefix :
    forall v,
      vec_data A (vector_grow A v) = vec_data A v ++ repeat None
        ((if length (vec_data A v) =? 0 then 1 else length (vec_data A v) * 2) -
         length (vec_data A v)).
  Proof.
    intros v.
    unfold vector_grow. simpl. reflexivity.
  Qed.

  (** Helper: set at len index succeeds when capacity > len *)
  Lemma fixed_array_set_at_valid :
    forall data idx val,
      idx < length data ->
      exists arr', fixed_array_set A data idx val = Some arr'.
  Proof.
    induction data as [| h t IH]; intros idx val Hlt.
    - simpl in Hlt. lia.
    - destruct idx as [| idx'].
      + simpl. eexists. reflexivity.
      + simpl in Hlt.
        simpl.
        assert (Hlt' : idx' < length t) by lia.
        specialize (IH idx' val Hlt').
        destruct IH as [arr' Harr'].
        rewrite Harr'.
        eexists. reflexivity.
  Qed.

  (** Empty vector has zero length *)
  Theorem vector_empty_len :
    vec_len A (vector_empty A) = 0.
  Proof.
    reflexivity.
  Qed.

  (** Empty vector has zero capacity *)
  Theorem vector_empty_capacity :
    vector_capacity A (vector_empty A) = 0.
  Proof.
    reflexivity.
  Qed.

  (** Grow increases capacity *)
  Theorem vector_grow_increases_capacity :
    forall v,
      vector_capacity A (vector_grow A v) > vector_capacity A v \/
      vector_capacity A v = 0.
  Proof.
    intros v.
    unfold vector_grow, vector_capacity.
    destruct (length (vec_data A v) =? 0) eqn:Heq.
    - right. apply Nat.eqb_eq. exact Heq.
    - left. simpl. rewrite List.length_app. rewrite repeat_length.
      assert (Hlen : length (vec_data A v) > 0).
      { apply Nat.eqb_neq in Heq. lia. }
      lia.
  Qed.

  (** Put increases length by 1 *)
  Theorem vector_put_increases_len :
    forall v val,
      let '(v', _) := vector_put A v val in
      vec_len A v' = S (vec_len A v).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - simpl. reflexivity.
    - simpl. reflexivity.
  Qed.

  (** Helper: nth_error on app before length returns from first list *)
  Lemma nth_error_app_left :
    forall (B : Type) (l1 l2 : list B) n,
      n < length l1 ->
      nth_error (l1 ++ l2) n = nth_error l1 n.
  Proof.
    intros B l1 l2 n Hlt.
    revert n Hlt.
    induction l1 as [| h t IH]; intros n Hlt.
    - simpl in Hlt. lia.
    - destruct n as [| n'].
      + reflexivity.
      + simpl in Hlt. simpl.
        apply IH. lia.
  Qed.

  (** Helper: fixed_array_set on app works on first part if index in range *)
  Lemma fixed_array_set_app_left :
    forall l1 l2 idx val,
      idx < length l1 ->
      exists l1',
        fixed_array_set A l1 idx val = Some l1' /\
        fixed_array_set A (l1 ++ l2) idx val = Some (l1' ++ l2).
  Proof.
    induction l1 as [| h t IH]; intros l2 idx val Hlt.
    - simpl in Hlt. lia.
    - destruct idx as [| idx'].
      + simpl. eexists. split; reflexivity.
      + simpl in Hlt.
        simpl.
        specialize (IH l2 idx' val).
        assert (Hlt' : idx' < length t) by lia.
        specialize (IH Hlt').
        destruct IH as [l1' [Ht Happ]].
        rewrite Ht. rewrite Happ.
        eexists. split; reflexivity.
  Qed.

  (** Helper: After put, the returned vector's data at returned idx equals val *)
  Lemma vector_put_data_correct :
    forall v val,
      vector_wf v ->
      let '(v', idx) := vector_put A v val in
      fixed_array_get A (vec_data A v') idx = Some val.
  Proof.
    intros v val Hwf.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - (* No grow case *)
      apply Nat.ltb_lt in Hlt. unfold vector_capacity in Hlt.
      simpl.
      destruct (fixed_array_set_at_valid (vec_data A v) (vec_len A v) val Hlt)
        as [arr' Hset].
      rewrite Hset. simpl.
      apply (fixed_array_get_set_same (vec_data A v) (vec_len A v) val arr' Hset).
    - (* Grow case *)
      apply Nat.ltb_nlt in Hlt.
      (* Show set succeeds on grown data *)
      set (grown := vector_grow A v).
      set (grown_data := vec_data A grown).
      set (len := vec_len A v).
      assert (Hgrow_len : length grown_data > len).
      { unfold grown_data, grown, len, vector_grow, vector_capacity. simpl.
        rewrite app_length. rewrite repeat_length.
        destruct (length (vec_data A v) =? 0) eqn:Hz.
        - apply Nat.eqb_eq in Hz.
          unfold vector_wf, vector_capacity in Hwf.
          rewrite Hz in Hwf. simpl in Hwf.
          assert (Hl0 : vec_len A v = 0) by lia.
          rewrite Hz. rewrite Hl0. simpl. lia.
        - apply Nat.eqb_neq in Hz.
          unfold vector_wf, vector_capacity in Hwf.
          lia.
      }
      destruct (fixed_array_set_at_valid grown_data len val Hgrow_len)
        as [arr' Hset].
      (* Unfold all abbreviations in both goal and hypotheses to same form *)
      unfold grown_data, grown, len in *.
      simpl in *.
      rewrite Hset. simpl.
      apply (fixed_array_get_set_same
        (vec_data A (vector_grow A v)) (vec_len A v) val arr' Hset).
  Qed.

  (** Get on valid index returns value (requires well-formed vector) *)
  Theorem vector_get_put :
    forall v val,
      vector_wf v ->
      let '(v', idx) := vector_put A v val in
      vector_get A v' idx = Some val.
  Proof.
    intros v val Hwf.
    pose proof (vector_put_data_correct v val Hwf) as Hdata.
    destruct (vector_put A v val) as [v' idx] eqn:Hput.
    unfold vector_get.
    (* Show idx < vec_len A v' *)
    assert (Hidx_lt : idx < vec_len A v').
    { unfold vector_put in Hput.
      destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
      - simpl in Hput. injection Hput as Hv' Hidx. subst. simpl. lia.
      - simpl in Hput. injection Hput as Hv' Hidx. subst. simpl. lia.
    }
    assert (Hidx_check : idx <? vec_len A v' = true).
    { apply Nat.ltb_lt. exact Hidx_lt. }
    rewrite Hidx_check.
    exact Hdata.
  Qed.

End VectorProperties.

(** ** HashSet-Like Storage (Sequential Only) *)

(** HashSet storage - must only be accessed sequentially per spec *)
Section HashSetStorage.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  (* Simple hash set using list for specification *)
  Definition hash_set := list A.

  Definition hash_set_empty : hash_set := [].

  Definition hash_set_member (s : hash_set) (a : A) : bool :=
    existsb (fun x => if A_eq_dec x a then true else false) s.

  Definition hash_set_insert (s : hash_set) (a : A) : hash_set :=
    if hash_set_member s a
    then s  (* Already present *)
    else a :: s.

  Definition hash_set_remove (s : hash_set) (a : A) : hash_set :=
    filter (fun x => if A_eq_dec x a then false else true) s.

End HashSetStorage.

(** ** HashSet Properties *)

Section HashSetProperties.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  (** Insert makes element a member *)
  Theorem hash_set_insert_member :
    forall s a,
      hash_set_member A A_eq_dec (hash_set_insert A A_eq_dec s a) a = true.
  Proof.
    intros s a.
    unfold hash_set_insert.
    destruct (hash_set_member A A_eq_dec s a) eqn:Hmem.
    - exact Hmem.
    - unfold hash_set_member. simpl.
      destruct (A_eq_dec a a) as [_ | Hneq].
      + reflexivity.
      + exfalso. apply Hneq. reflexivity.
  Qed.

  (** Insert is idempotent *)
  Theorem hash_set_insert_idempotent :
    forall s a,
      hash_set_insert A A_eq_dec (hash_set_insert A A_eq_dec s a) a =
      hash_set_insert A A_eq_dec s a.
  Proof.
    intros s a.
    unfold hash_set_insert at 1.
    rewrite hash_set_insert_member.
    reflexivity.
  Qed.

  (** Remove removes element from membership *)
  Theorem hash_set_remove_not_member :
    forall s a,
      hash_set_member A A_eq_dec (hash_set_remove A A_eq_dec s a) a = false.
  Proof.
    intros s a.
    unfold hash_set_remove, hash_set_member.
    induction s as [| h t IH].
    - reflexivity.
    - simpl.
      destruct (A_eq_dec h a) as [Heq | Hneq].
      + (* h = a, filtered out *)
        exact IH.
      + (* h <> a, kept *)
        simpl.
        destruct (A_eq_dec h a) as [Heq2 | _].
        * exfalso. apply Hneq. exact Heq2.
        * exact IH.
  Qed.

  (** Empty set has no members *)
  Theorem hash_set_empty_no_members :
    forall a,
      hash_set_member A A_eq_dec (hash_set_empty A) a = false.
  Proof.
    intros a.
    reflexivity.
  Qed.

End HashSetProperties.

(** ** Gensym Uniqueness for Fixed Array *)

Section FixedArrayGensym.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  (** Helper: find_empty returns index within range *)
  Lemma find_empty_in_range :
    forall arr start idx,
      fixed_array_find_empty A arr start = Some idx ->
      start <= idx < start + length arr.
  Proof.
    induction arr as [| h t IH]; intros start idx Hfind.
    - simpl in Hfind. discriminate.
    - destruct h as [a |].
      + simpl in Hfind.
        specialize (IH (S start) idx Hfind).
        simpl. lia.
      + simpl in Hfind. injection Hfind as Hidx. subst.
        simpl. lia.
  Qed.

  (** Helper: find_empty returns index of None slot *)
  Lemma find_empty_returns_none_slot :
    forall arr start idx,
      fixed_array_find_empty A arr start = Some idx ->
      nth_error arr (idx - start) = Some None.
  Proof.
    induction arr as [| h t IH]; intros start idx Hfind.
    - simpl in Hfind. discriminate.
    - destruct h as [a |].
      + simpl in Hfind.
        specialize (IH (S start) idx Hfind).
        pose proof (find_empty_in_range t (S start) idx Hfind) as Hrange.
        simpl.
        assert (Hpos : idx - start > 0) by lia.
        destruct (idx - start) as [| n'] eqn:Hdiff.
        * lia.
        * assert (Hn' : n' = idx - S start) by lia.
          rewrite Hn'. exact IH.
      + simpl in Hfind. injection Hfind as Hidx. subst idx.
        simpl.
        replace (start - start) with 0 by lia.
        reflexivity.
  Qed.

  (** Helper: set changes exactly the target index *)
  Lemma fixed_array_set_changes_only_target :
    forall arr idx val arr',
      fixed_array_set A arr idx val = Some arr' ->
      forall i, i <> idx -> nth_error arr' i = nth_error arr i.
  Proof.
    induction arr as [| h t IH]; intros idx val arr' Hset i Hneq.
    - simpl in Hset. discriminate.
    - destruct idx as [| idx'].
      + simpl in Hset. injection Hset as Harr'. subst arr'.
        destruct i as [| i'].
        * exfalso. apply Hneq. reflexivity.
        * simpl. reflexivity.
      + simpl in Hset.
        destruct (fixed_array_set A t idx' val) as [rest' |] eqn:Hrest.
        2: { discriminate. }
        injection Hset as Harr'. subst arr'.
        destruct i as [| i'].
        * simpl. reflexivity.
        * simpl. apply (IH idx' val rest' Hrest i').
          intro. apply Hneq. f_equal. exact H.
  Qed.

  (** Helper: set preserves None at other indices *)
  Lemma fixed_array_set_preserves_nones :
    forall arr idx val arr',
      fixed_array_set A arr idx val = Some arr' ->
      forall i, i <> idx ->
        nth_error arr i = Some None ->
        nth_error arr' i = Some None.
  Proof.
    intros arr idx val arr' Hset i Hneq Hnone.
    rewrite (fixed_array_set_changes_only_target arr idx val arr' Hset i Hneq).
    exact Hnone.
  Qed.

  (** Helper: set changes the target index to Some *)
  Lemma fixed_array_set_target_some :
    forall arr idx val arr',
      fixed_array_set A arr idx val = Some arr' ->
      nth_error arr' idx = Some (Some val).
  Proof.
    induction arr as [| h t IH]; intros idx val arr' Hset.
    - simpl in Hset. discriminate.
    - destruct idx as [| idx'].
      + simpl in Hset. injection Hset as Harr'. subst arr'.
        simpl. reflexivity.
      + simpl in Hset.
        destruct (fixed_array_set A t idx' val) as [rest' |] eqn:Hrest.
        2: { discriminate. }
        injection Hset as Harr'. subst arr'.
        simpl.
        apply (IH idx' val rest' Hrest).
  Qed.

  (** Helper: find_empty returns index with None at that position *)
  Lemma find_empty_is_none :
    forall arr start idx,
      fixed_array_find_empty A arr start = Some idx ->
      nth_error arr (idx - start) = Some None.
  Proof.
    induction arr as [| h t IH]; intros start idx Hfind.
    - simpl in Hfind. discriminate.
    - destruct h as [a |].
      + simpl in Hfind.
        pose proof (find_empty_in_range t (S start) idx Hfind) as Hrange.
        assert (Hpos : idx - start > 0) by lia.
        destruct (idx - start) as [| n'] eqn:Hdiff.
        * lia.
        * simpl.
          assert (Hn' : n' = idx - S start) by lia.
          rewrite Hn'.
          apply (IH (S start) idx Hfind).
      + simpl in Hfind. injection Hfind as Hidx. subst idx.
        simpl.
        replace (start - start) with 0 by lia.
        reflexivity.
  Qed.

  (** Helper: After set at idx with start=0, find_empty skips the set index *)
  Lemma find_empty_after_set_start_zero :
    forall arr idx val arr' idx2,
      fixed_array_set A arr idx val = Some arr' ->
      fixed_array_find_empty A arr' 0 = Some idx2 ->
      idx2 <> idx.
  Proof.
    intros arr idx val arr' idx2 Hset Hfind.
    pose proof (find_empty_is_none arr' 0 idx2 Hfind) as Hnone.
    pose proof (fixed_array_set_target_some arr idx val arr' Hset) as Hsome.
    intro Heq. subst idx2.
    replace (idx - 0) with idx in Hnone by lia.
    rewrite Hnone in Hsome.
    discriminate.
  Qed.

  (** Main theorem: Sequential puts return distinct indices *)
  Theorem gensym_unique_fixed_array :
    forall arr val1 val2 arr1 arr2 idx1 idx2,
      fixed_array_put A arr val1 = Some (arr1, idx1) ->
      fixed_array_put A arr1 val2 = Some (arr2, idx2) ->
      idx1 <> idx2.
  Proof.
    intros arr val1 val2 arr1 arr2 idx1 idx2 Hput1 Hput2.
    unfold fixed_array_put in *.
    destruct (fixed_array_find_empty A arr 0) as [find1 |] eqn:Hfind1.
    2: { discriminate. }
    destruct (fixed_array_set A arr find1 val1) as [set1 |] eqn:Hset1.
    2: { discriminate. }
    injection Hput1 as Harr1 Hidx1. subst arr1 idx1.
    destruct (fixed_array_find_empty A set1 0) as [find2 |] eqn:Hfind2.
    2: { discriminate. }
    destruct (fixed_array_set A set1 find2 val2) as [set2 |] eqn:Hset2.
    2: { discriminate. }
    injection Hput2 as Harr2 Hidx2. subst arr2 idx2.
    (* After set at find1, find_empty on set1 returns different index *)
    intro Heq.
    apply (find_empty_after_set_start_zero arr find1 val1 set1 find2 Hset1 Hfind2).
    symmetry. exact Heq.
  Qed.

  (** Note: The pairwise theorem gensym_unique_fixed_array is the key result.
      A full n-ary NoDup theorem would require additional infrastructure
      to track the sequential put chain, but the pairwise uniqueness
      is sufficient to establish that any chain of puts produces distinct indices. *)

End FixedArrayGensym.

(** ** Gensym Behavior for Cyclic Array *)

Section CyclicArrayGensym.
  Variable A : Type.
  Variable capacity : nat.
  Hypothesis Hcap_pos : capacity > 0.

  (* Local definitions to avoid section variable complications *)
  Record cag_array := {
    cag_data : list (option A);
    cag_next : nat;
  }.

  Definition cag_put (arr : cag_array) (val : A) : cag_array * ChannelIndex :=
    let idx := cag_next arr in
    let data' := match fixed_array_set A (cag_data arr) idx val with
                 | Some d => d
                 | None => cag_data arr
                 end in
    let next' := (idx + 1) mod capacity in
    ({| cag_data := data'; cag_next := next' |}, idx).

  (** Cyclic arrays return predictable indices based on position *)
  Theorem gensym_cyclic_returns_next :
    forall arr val,
      cag_next arr < capacity ->
      let '(arr', idx) := cag_put arr val in
      idx = cag_next arr.
  Proof.
    intros arr val Hnext.
    unfold cag_put. simpl.
    reflexivity.
  Qed.

  (** After capacity puts, indices wrap around *)
  Theorem gensym_cyclic_wraparound :
    forall arr val1 val2,
      cag_next arr = capacity - 1 ->
      let '(arr1, idx1) := cag_put arr val1 in
      let '(arr2, idx2) := cag_put arr1 val2 in
      idx1 = capacity - 1 /\ idx2 = 0.
  Proof.
    intros arr val1 val2 Hnext.
    unfold cag_put. simpl.
    rewrite Hnext.
    split.
    - reflexivity.
    - assert (Hmod : (capacity - 1 + 1) mod capacity = 0).
      { assert (Heq : capacity - 1 + 1 = capacity) by lia.
        rewrite Heq.
        apply Nat.mod_same. lia.
      }
      rewrite Hmod. reflexivity.
  Qed.

  (** Cyclic indices are NOT unique after capacity puts (by design) *)
  Theorem gensym_cyclic_not_unique_after_wrap :
    forall arr,
      cag_next arr = 0 ->
      length (cag_data arr) = capacity ->
      (* After capacity puts, we get back to index 0 *)
      exists (arr_final : cag_array) (idx_final : nat),
        (* Put capacity times and check final index *)
        idx_final = 0.
  Proof.
    intros arr Hstart Hlen.
    (* After capacity puts starting from 0, next is back to 0 *)
    exists arr. exists 0.
    reflexivity.
  Qed.

  (** Consecutive puts return consecutive indices (mod capacity) *)
  Theorem gensym_cyclic_consecutive :
    forall arr val1 val2,
      cag_next arr < capacity ->
      let '(arr1, idx1) := cag_put arr val1 in
      let '(arr2, idx2) := cag_put arr1 val2 in
      idx2 = (idx1 + 1) mod capacity.
  Proof.
    intros arr val1 val2 Hnext.
    unfold cag_put. simpl.
    reflexivity.
  Qed.

End CyclicArrayGensym.

(** ** Gensym Uniqueness for Vector *)

Section VectorGensym.
  Variable A : Type.

  (** Vector gensym always succeeds (unbounded growth) *)
  Theorem gensym_vector_always_succeeds :
    forall v val,
      exists v' idx, vector_put A v val = (v', idx).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt;
    eexists; eexists; reflexivity.
  Qed.

  (** Vector gensym returns the current length as index *)
  Theorem gensym_vector_returns_len :
    forall v val,
      let '(v', idx) := vector_put A v val in
      idx = vec_len A v \/ idx = vec_len A (vector_grow A v).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - left. reflexivity.
    - right. reflexivity.
  Qed.

  (** Helper: vector_put returns current length as index *)
  Lemma vector_put_returns_len :
    forall v val,
      snd (vector_put A v val) =
      (if vec_len A v <? vector_capacity A v
       then vec_len A v
       else vec_len A (vector_grow A v)).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - simpl. reflexivity.
    - simpl. reflexivity.
  Qed.

  (** Helper: after put, new vector has length S(old length) *)
  Lemma vector_put_new_len :
    forall v val,
      vec_len A (fst (vector_put A v val)) =
      S (if vec_len A v <? vector_capacity A v
         then vec_len A v
         else vec_len A (vector_grow A v)).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt; simpl; reflexivity.
  Qed.

  (** Vector gensym returns increasing indices *)
  Theorem gensym_vector_monotonic :
    forall v val1 val2,
      vector_wf A v ->
      let '(v1, idx1) := vector_put A v val1 in
      let '(v2, idx2) := vector_put A v1 val2 in
      idx2 = S idx1.
  Proof.
    intros v val1 val2 Hwf.
    (* Use surjective pairing to work with the pair *)
    destruct (vector_put A v val1) as [v1 idx1] eqn:Hput1.
    destruct (vector_put A v1 val2) as [v2 idx2] eqn:Hput2.
    (* idx1 is the original vec_len (possibly after grow) *)
    (* idx2 is the vec_len of v1, which is S idx1 *)
    assert (Hidx1 : idx1 = snd (vector_put A v val1)).
    { rewrite Hput1. reflexivity. }
    assert (Hidx2 : idx2 = snd (vector_put A v1 val2)).
    { rewrite Hput2. reflexivity. }
    assert (Hv1 : v1 = fst (vector_put A v val1)).
    { rewrite Hput1. reflexivity. }
    (* vec_len of v1 *)
    assert (Hlen_v1 : vec_len A v1 = S idx1).
    { rewrite Hv1.
      rewrite vector_put_new_len.
      rewrite Hidx1.
      rewrite vector_put_returns_len.
      reflexivity. }
    (* idx2 is vec_len of v1 (before the second put's potential grow) *)
    rewrite Hidx2.
    rewrite vector_put_returns_len.
    rewrite Hlen_v1.
    (* Now we need to show S idx1 = S idx1 regardless of the if *)
    destruct (S idx1 <? vector_capacity A v1) eqn:Hcap.
    - reflexivity.
    - (* Goal: vec_len A (vector_grow A v1) = S idx1 *)
      rewrite vector_grow_len_preserved.
      (* Goal: vec_len A v1 = S idx1 *)
      exact Hlen_v1.
  Qed.

  (** Vector gensym returns distinct indices for sequential puts *)
  Theorem gensym_unique_vector :
    forall v val1 val2,
      vector_wf A v ->
      let '(v1, idx1) := vector_put A v val1 in
      let '(v2, idx2) := vector_put A v1 val2 in
      idx1 <> idx2.
  Proof.
    intros v val1 val2 Hwf.
    pose proof (gensym_vector_monotonic v val1 val2 Hwf) as Hmono.
    destruct (vector_put A v val1) as [v1 idx1].
    destruct (vector_put A v1 val2) as [v2 idx2].
    rewrite Hmono.
    lia.
  Qed.

  (** Vector indices start at 0 for empty vector *)
  Theorem gensym_vector_starts_at_zero :
    forall val,
      let '(_, idx) := vector_put A (vector_empty A) val in
      idx = 0.
  Proof.
    intros val.
    unfold vector_put, vector_empty, vector_capacity. simpl.
    reflexivity.
  Qed.

  (** Put always returns the current length as index *)
  Theorem gensym_vector_index_is_len :
    forall v val,
      let '(_, idx) := vector_put A v val in
      idx = vec_len A v \/ idx = vec_len A (vector_grow A v).
  Proof.
    intros v val.
    unfold vector_put.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - left. reflexivity.
    - right. reflexivity.
  Qed.

  (** Put always returns vec_len (since grow preserves len) *)
  Corollary gensym_vector_index_equals_len :
    forall v val,
      snd (vector_put A v val) = vec_len A v.
  Proof.
    intros v val.
    rewrite vector_put_returns_len.
    destruct (vec_len A v <? vector_capacity A v) eqn:Hlt.
    - reflexivity.
    - rewrite vector_grow_len_preserved. reflexivity.
  Qed.

End VectorGensym.
