(** * VectorDB Collection Specification

    This module specifies the behavior of VectorDB collections for
    similarity-based matching. VectorDB stores embedding vectors and
    supports k-nearest neighbors queries based on cosine similarity.

    Reference: Spec "Reifying RSpaces.md" lines 234-240
*)

From Coq Require Import List Bool ZArith Lia QArith Qabs.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Embedding Vector Type *)

(** Embedding vectors use rationals for precise arithmetic specification *)
Definition Embedding := list Q.

(** Strict less-than for Q (not in standard library) *)
Definition Qlt_bool (x y : Q) : bool := negb (Qle_bool y x).

(** Helper lemma: sum of non-negative rationals is non-negative *)
Lemma Qplus_nonneg : forall a b : Q, 0 <= a -> 0 <= b -> 0 <= a + b.
Proof.
  intros a b Ha Hb.
  setoid_replace 0 with (0 + 0) by (symmetry; apply Qplus_0_l).
  apply Qplus_le_compat; assumption.
Qed.

(** Helper lemma: square of a rational is non-negative *)
Lemma Qsquare_nonneg : forall x : Q, 0 <= x * x.
Proof.
  intros x.
  destruct (Qlt_le_dec x 0) as [Hneg | Hpos].
  - (* x < 0: then -x > 0, and x * x = (-x) * (-x) *)
    setoid_replace (x * x) with ((- x) * (- x)) by ring.
    apply Qlt_le_weak.
    apply Qmult_lt_0_compat.
    + (* 0 < -x: from x < 0 we get -0 < -x, and -0 == 0 *)
      setoid_replace 0 with (- 0) by ring.
      apply Qopp_lt_compat. exact Hneg.
    + setoid_replace 0 with (- 0) by ring.
      apply Qopp_lt_compat. exact Hneg.
  - (* 0 <= x: use Qmult_le_0_compat *)
    apply Qmult_le_0_compat; exact Hpos.
Qed.

(** ** Vector Operations *)

Section VectorOperations.

  (** Dot product of two vectors *)
  Fixpoint dot_product (v1 v2 : Embedding) : Q :=
    match v1, v2 with
    | [], _ => 0
    | _, [] => 0
    | x :: xs, y :: ys => (x * y) + dot_product xs ys
    end.

  (** Squared magnitude (to avoid square root) *)
  Fixpoint magnitude_squared (v : Embedding) : Q :=
    match v with
    | [] => 0
    | x :: xs => (x * x) + magnitude_squared xs
    end.

  (** Cosine similarity: dot(v1, v2) / (|v1| * |v2|)
      We use squared magnitudes to avoid square roots.
      The comparison works because we compare against threshold^2 * |v1|^2 * |v2|^2 *)
  Definition cosine_similarity_scaled (v1 v2 : Embedding) : Q * Q :=
    let dp := dot_product v1 v2 in
    let m1_sq := magnitude_squared v1 in
    let m2_sq := magnitude_squared v2 in
    (dp * dp, m1_sq * m2_sq).  (* cos^2 numerator, denominator *)

  (** Compare cosine similarity against threshold (using squared values) *)
  Definition cosine_above_threshold (v1 v2 : Embedding) (threshold_sq : Q) : bool :=
    let '(cos_sq_num, denom) := cosine_similarity_scaled v1 v2 in
    (* cos^2 >= threshold^2 iff cos_sq_num / denom >= threshold_sq
       iff cos_sq_num >= threshold_sq * denom (assuming positive) *)
    Qle_bool (threshold_sq * denom) cos_sq_num.

End VectorOperations.

(** ** VectorDB Storage *)

Section VectorDBStorage.

  (** VectorDB configuration *)
  Record VectorDBConfig := {
    vdb_similarity_threshold_sq : Q;  (* Squared threshold for efficiency *)
    vdb_k_neighbors : nat;            (* Maximum neighbors to return *)
  }.

  (** Stored embedding with associated data *)
  Record StoredEmbedding (A : Type) := {
    se_embedding : Embedding;
    se_data : A;
    se_persist : bool;
  }.

  Arguments se_embedding {A}.
  Arguments se_data {A}.
  Arguments se_persist {A}.

  (** VectorDB storage *)
  Definition VectorDB (A : Type) := list (StoredEmbedding A).

  Definition vectordb_empty (A : Type) : VectorDB A := [].

  (** Put an embedding with data *)
  Definition vectordb_put {A : Type} (db : VectorDB A) (emb : Embedding) (data : A) (persist : bool)
    : VectorDB A :=
    {| se_embedding := emb; se_data := data; se_persist := persist |} :: db.

  (** Compute similarity score between query and stored embedding *)
  Definition similarity_score (query : Embedding) (stored : Embedding) : Q * Q :=
    cosine_similarity_scaled query stored.

  (** Check if embedding is above threshold *)
  Definition is_similar (query stored : Embedding) (config : VectorDBConfig) : bool :=
    cosine_above_threshold query stored (vdb_similarity_threshold_sq config).

  (** Find all embeddings above similarity threshold *)
  Definition vectordb_find_similar {A : Type} (db : VectorDB A) (query : Embedding) (config : VectorDBConfig)
    : list (StoredEmbedding A * (Q * Q)) :=
    filter (fun '(se, _) => is_similar query (se_embedding se) config)
           (map (fun se => (se, similarity_score query (se_embedding se))) db).

  (** Sort by similarity (descending) - simplified using score comparison *)
  Fixpoint insert_sorted {A : Type} (item : StoredEmbedding A * (Q * Q))
                                     (sorted : list (StoredEmbedding A * (Q * Q)))
    : list (StoredEmbedding A * (Q * Q)) :=
    match sorted with
    | [] => [item]
    | h :: t =>
      let '(_, (num1, denom1)) := item in
      let '(_, (num2, denom2)) := h in
      (* Compare: num1/denom1 > num2/denom2 iff num1*denom2 > num2*denom1 *)
      if Qlt_bool (num2 * denom1) (num1 * denom2)
      then item :: h :: t
      else h :: insert_sorted item t
    end.

  Fixpoint sort_by_similarity {A : Type} (items : list (StoredEmbedding A * (Q * Q)))
    : list (StoredEmbedding A * (Q * Q)) :=
    match items with
    | [] => []
    | h :: t => insert_sorted h (sort_by_similarity t)
    end.

  (** Take first k elements *)
  Fixpoint take_k {B : Type} (k : nat) (xs : list B) : list B :=
    match k, xs with
    | O, _ => []
    | _, [] => []
    | S k', x :: xs' => x :: take_k k' xs'
    end.

  (** K-nearest neighbors query *)
  Definition vectordb_knn {A : Type} (db : VectorDB A) (query : Embedding) (config : VectorDBConfig)
    : list (StoredEmbedding A) :=
    let similar := vectordb_find_similar db query config in
    let sorted := sort_by_similarity similar in
    let top_k := take_k (vdb_k_neighbors config) sorted in
    map fst top_k.

  (** Find and remove matching embedding (for consume) *)
  Fixpoint vectordb_find_and_remove {A : Type} (db : VectorDB A) (query : Embedding) (config : VectorDBConfig)
    : option (StoredEmbedding A * VectorDB A) :=
    match db with
    | [] => None
    | se :: rest =>
      if is_similar query (se_embedding se) config
      then
        if se_persist se
        then Some (se, db)  (* Keep persistent *)
        else Some (se, rest)  (* Remove non-persistent *)
      else
        match vectordb_find_and_remove rest query config with
        | None => None
        | Some (found, rest') => Some (found, se :: rest')
        end
    end.

End VectorDBStorage.

(** ** VectorDB Properties *)

Section VectorDBProperties.
  Variable A : Type.

  (** Dot product is commutative *)
  Theorem dot_product_comm :
    forall v1 v2,
      length v1 = length v2 ->
      dot_product v1 v2 == dot_product v2 v1.
  Proof.
    induction v1 as [| x xs IH]; intros v2 Hlen.
    - destruct v2; simpl; [reflexivity | simpl in Hlen; discriminate].
    - destruct v2 as [| y ys]; simpl in Hlen.
      + discriminate.
      + simpl. injection Hlen as Hlen'.
        specialize (IH ys Hlen').
        rewrite Qmult_comm.
        rewrite IH.
        reflexivity.
  Qed.

  (** Dot product with zero vector is zero *)
  Theorem dot_product_zero_left :
    forall v,
      dot_product (repeat 0 (length v)) v == 0.
  Proof.
    induction v as [| x xs IH].
    - reflexivity.
    - simpl. rewrite Qmult_0_l. rewrite Qplus_0_l.
      simpl in IH. exact IH.
  Qed.

  (** Magnitude squared is non-negative *)
  Theorem magnitude_squared_nonneg :
    forall v,
      0 <= magnitude_squared v.
  Proof.
    induction v as [| x xs IH].
    - simpl. apply Qle_refl.
    - simpl.
      apply Qplus_nonneg.
      + apply Qsquare_nonneg.
      + exact IH.
  Qed.

  (** Empty database has no similar elements *)
  Theorem vectordb_empty_no_similar :
    forall query config,
      vectordb_find_similar (vectordb_empty A) query config = [].
  Proof.
    intros. reflexivity.
  Qed.

  (** Put increases database size by 1 *)
  Theorem vectordb_put_length :
    forall db emb data persist,
      length (@vectordb_put A db emb data persist) = S (length db).
  Proof.
    intros. reflexivity.
  Qed.

  (** KNN returns at most k elements *)
  Theorem vectordb_knn_at_most_k :
    forall db query config,
      (length (@vectordb_knn A db query config) <= vdb_k_neighbors config)%nat.
  Proof.
    intros db query config.
    unfold vectordb_knn.
    (* take_k returns at most k elements *)
    generalize (@sort_by_similarity A (@vectordb_find_similar A db query config)).
    intros sorted.
    generalize (vdb_k_neighbors config) as k.
    induction sorted as [| h t IH]; intros k.
    - simpl. destruct k; simpl; lia.
    - simpl. destruct k.
      + simpl. lia.
      + simpl. specialize (IH k). lia.
  Qed.

  (** Find and remove returns element from database *)
  Theorem vectordb_find_and_remove_in_db :
    forall db query config se rest,
      @vectordb_find_and_remove A db query config = Some (se, rest) ->
      In se db.
  Proof.
    induction db as [| h t IH]; intros query config se rest Hfind.
    - simpl in Hfind. discriminate.
    - simpl in Hfind.
      destruct (is_similar query (@se_embedding A h) config) eqn:Hsim.
      + destruct (@se_persist A h) eqn:Hpers.
        * injection Hfind as Hse Hrest. subst.
          left. reflexivity.
        * injection Hfind as Hse Hrest. subst.
          left. reflexivity.
      + destruct (@vectordb_find_and_remove A t query config) as [[found rest']|] eqn:Hrec.
        * injection Hfind as Hse Hrest. subst.
          right. apply (IH query config se rest' Hrec).
        * discriminate.
  Qed.

  (** Similar elements are actually similar (above threshold) *)
  Theorem vectordb_find_similar_are_similar :
    forall db query config se score,
      In (se, score) (@vectordb_find_similar A db query config) ->
      is_similar query (@se_embedding A se) config = true.
  Proof.
    intros db query config se score Hin.
    unfold vectordb_find_similar in Hin.
    apply filter_In in Hin.
    destruct Hin as [_ Hsim].
    exact Hsim.
  Qed.

End VectorDBProperties.

(** ** Integration with Match Trait *)

Section VectorDBMatch.

  (** VectorDB matcher using similarity *)
  Definition vectordb_match (config : VectorDBConfig) (pattern data : Embedding) : bool :=
    is_similar pattern data config.

  (** Match is symmetric for equal-length vectors *)
  Theorem vectordb_match_symmetric :
    forall config pattern data,
      length pattern = length data ->
      vectordb_match config pattern data = vectordb_match config data pattern.
  Proof.
    intros config pattern data Hlen.
    unfold vectordb_match, is_similar, cosine_above_threshold, cosine_similarity_scaled.
    (* Dot product is commutative, magnitude_squared is the same for both *)
    assert (Hdp : dot_product pattern data == dot_product data pattern).
    { apply dot_product_comm. exact Hlen. }
    (* Use Qleb_comp to show Qle_bool respects Qeq *)
    apply Qleb_comp.
    - (* threshold * (m_pattern * m_data) == threshold * (m_data * m_pattern) *)
      apply Qmult_comp. reflexivity.
      apply Qmult_comm.
    - (* dp_pattern_data^2 == dp_data_pattern^2 *)
      apply Qmult_comp; exact Hdp.
  Qed.

End VectorDBMatch.
