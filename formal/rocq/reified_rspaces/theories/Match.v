(** * Match Trait Specification

    This module specifies the Match<P, A> trait for pluggable pattern matching
    strategies in reified RSpaces.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/spaces/matcher.rs
*)

From Coq Require Import List Bool String.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.
Open Scope string_scope.

(** ** Match Trait *)

(** Abstract pattern matching interface.
    P is the pattern type, A is the data type. *)
Class Match (P A : Type) := {
  (** Returns true if pattern matches data *)
  match_fn : P -> A -> bool;

  (** Matcher name for debugging *)
  matcher_name : string;
}.

(** ** Match Properties *)

Section MatchProperties.
  Context {P A : Type} `{M : Match P A}.

  (** Reflexivity: A pattern that matches everything *)
  Definition match_reflexive (p : P) : Prop :=
    forall a, match_fn p a = true.

  (** A wildcard pattern matches all data *)
  Definition is_wildcard (p : P) : Prop := match_reflexive p.

  (** Pattern refinement: p1 is more specific than p2 *)
  Definition pattern_refines (p1 p2 : P) : Prop :=
    forall a, match_fn p1 a = true -> match_fn p2 a = true.

  (** Match is deterministic *)
  Definition match_deterministic : Prop :=
    forall p a, match_fn p a = true \/ match_fn p a = false.

  (** Match respects boolean decidability *)
  Lemma match_decidable : forall p a, {match_fn p a = true} + {match_fn p a = false}.
  Proof.
    intros p a.
    destruct (match_fn p a) eqn:Hm.
    - left. reflexivity.
    - right. reflexivity.
  Qed.
End MatchProperties.

(** ** Exact Match *)

(** Exact match requires equality between pattern and data. *)
Section ExactMatch.
  Variable A : Type.
  Variable A_eq_dec : forall a1 a2 : A, {a1 = a2} + {a1 <> a2}.

  Definition exact_match_fn (pattern data : A) : bool :=
    if A_eq_dec pattern data then true else false.

  Global Instance ExactMatcher : Match A A := {
    match_fn := exact_match_fn;
    matcher_name := "ExactMatch";
  }.

  (** Exact match is reflexive on equal values *)
  Theorem exact_match_reflexive : forall a, exact_match_fn a a = true.
  Proof.
    intros a. unfold exact_match_fn.
    destruct (A_eq_dec a a).
    - reflexivity.
    - exfalso. apply n. reflexivity.
  Qed.

  (** Exact match is symmetric *)
  Theorem exact_match_symmetric :
    forall a1 a2, exact_match_fn a1 a2 = true -> exact_match_fn a2 a1 = true.
  Proof.
    intros a1 a2 H.
    unfold exact_match_fn in *.
    destruct (A_eq_dec a1 a2).
    - rewrite e. destruct (A_eq_dec a2 a2).
      + reflexivity.
      + exfalso. apply n. reflexivity.
    - discriminate.
  Qed.
End ExactMatch.

(** ** Wildcard Match *)

(** Wildcard match accepts any data. *)
Section WildcardMatch.
  Variable P A : Type.

  Definition wildcard_match_fn (pattern : P) (data : A) : bool := true.

  Global Instance WildcardMatcher : Match P A := {
    match_fn := wildcard_match_fn;
    matcher_name := "WildcardMatch";
  }.

  (** Wildcard matches everything *)
  Theorem wildcard_always_matches : forall p a, wildcard_match_fn p a = true.
  Proof.
    intros. reflexivity.
  Qed.

  (** Every wildcard pattern is reflexive *)
  Theorem wildcard_is_reflexive : forall p, is_wildcard p.
  Proof.
    intros p a. reflexivity.
  Qed.
End WildcardMatch.

(** ** Match Combinators *)

Section MatchCombinators.
  Variable P A : Type.
  Variable M1 M2 : Match P A.

  (** And combinator: both matchers must match *)
  Definition and_match_fn (p : P) (a : A) : bool :=
    @match_fn P A M1 p a && @match_fn P A M2 p a.

  (** Or combinator: either matcher can match *)
  Definition or_match_fn (p : P) (a : A) : bool :=
    @match_fn P A M1 p a || @match_fn P A M2 p a.

  (** And match properties *)
  Lemma and_match_implies_both :
    forall p a, and_match_fn p a = true ->
    @match_fn P A M1 p a = true /\ @match_fn P A M2 p a = true.
  Proof.
    intros p a H.
    unfold and_match_fn in H.
    apply andb_prop in H.
    exact H.
  Qed.

  (** Or match properties *)
  Lemma or_match_implies_either :
    forall p a, or_match_fn p a = true ->
    @match_fn P A M1 p a = true \/ @match_fn P A M2 p a = true.
  Proof.
    intros p a H.
    unfold or_match_fn in H.
    apply orb_prop in H.
    exact H.
  Qed.
End MatchCombinators.

(** ** Match Substitution *)

(** If two data values are equivalent under all patterns,
    then a match on one implies a match on the other. *)
Section MatchSubstitution.
  Context {P A : Type} `{M : Match P A}.

  Definition data_equivalent (a1 a2 : A) : Prop :=
    forall p, match_fn p a1 = match_fn p a2.

  Theorem match_substitution :
    forall p a1 a2,
      match_fn p a1 = true ->
      data_equivalent a1 a2 ->
      match_fn p a2 = true.
  Proof.
    intros p a1 a2 Hmatch Heq.
    unfold data_equivalent in Heq.
    rewrite <- Heq.
    exact Hmatch.
  Qed.
End MatchSubstitution.

(** ** Integration with Data Collections *)

(** Match-based find operation for collections. *)
Section MatchWithCollections.
  Variable A : Type.
  Variable M : Match A A.

  (** Find first matching element in a list *)
  Fixpoint find_with_matcher (pattern : A) (l : list (A * bool))
    : option (A * list (A * bool)) :=
    match l with
    | [] => None
    | (data, persist) :: rest =>
      if @match_fn A A M pattern data then
        if persist then Some (data, l)
        else Some (data, rest)
      else
        match find_with_matcher pattern rest with
        | None => None
        | Some (found, rest') => Some (found, (data, persist) :: rest')
        end
    end.

  (** If find succeeds, the result actually matches the pattern *)
  Theorem find_with_matcher_correct :
    forall pattern l data rest,
      find_with_matcher pattern l = Some (data, rest) ->
      @match_fn A A M pattern data = true.
  Proof.
    intros pattern l. induction l as [| [d p] l' IH].
    - intros data rest H. discriminate.
    - intros data rest H. simpl in H.
      destruct (@match_fn A A M pattern d) eqn:Hmatch.
      + destruct p.
        * injection H as Hd _. rewrite <- Hd. exact Hmatch.
        * injection H as Hd _. rewrite <- Hd. exact Hmatch.
      + destruct (find_with_matcher pattern l') eqn:Hfind.
        * destruct p0 as [found rest']. injection H as Hd Hr.
          rewrite <- Hd. apply IH with rest'. reflexivity.
        * discriminate.
  Qed.
End MatchWithCollections.
