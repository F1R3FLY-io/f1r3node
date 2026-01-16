//! Property-Based Tests for Pattern Matching Module
//!
//! This module provides comprehensive property-based testing for the Match trait
//! and its various implementations in the Reified RSpaces system.
//!
//! # Formal Correspondence
//! - `Match.v`: Pattern matching properties (reflexivity, symmetry, monotonicity)
//! - `Collections/VectorDB.v`: Vector similarity semantics
//! - `Safety/Properties.v`: Matching safety guarantees
//!
//! # Test Coverage
//! - ExactMatch: Reflexivity, symmetry, transitivity
//! - WildcardMatch: Universal matching property
//! - VectorDBMatch: Threshold semantics, cosine similarity range
//! - AndMatch: Conjunction semantics
//! - OrMatch: Disjunction semantics
//! - Per-pattern threshold override
//!
//! # Rholang Syntax Examples
//!
//! ```rholang
//! // Exact matching for structural patterns
//! for (@{42} <- ch) { ... }  // Only matches exact value
//!
//! // Wildcard matching
//! for (_ <- ch) { ... }  // Matches any value
//!
//! // VectorDB similarity matching with threshold
//! new VectorSpace(`rho:space:queue:vectordb:default`) in {
//!   VectorSpace!({}, *space) |
//!   use space {
//!     for (@embedding <= pattern[threshold=0.9]) { ... }
//!   }
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;
use std::fmt::Debug;

use rholang::rust::interpreter::spaces::{
    Match, ExactMatch, WildcardMatch, VectorDBMatch, VectorPattern,
    AndMatch, OrMatch, PredicateMatcher, boxed,
};

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 500;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Arbitrary Generators
// =============================================================================

/// Generate a normalized f64 vector of the given dimensions.
fn arb_normalized_vector(dims: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<f64>> {
    prop_vec(-1.0f64..1.0f64, dims).prop_map(|v| {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            // Return unit vector if zero
            let len = v.len().max(1);
            let mut unit = vec![0.0; len];
            if !unit.is_empty() {
                unit[0] = 1.0;
            }
            unit
        }
    })
}

/// Generate an arbitrary f64 vector (not necessarily normalized).
fn arb_f64_vector(dims: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<f64>> {
    prop_vec(-100.0f64..100.0f64, dims)
}

/// Generate an arbitrary f32 vector.
fn arb_f32_vector(dims: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<f32>> {
    prop_vec(-100.0f32..100.0f32, dims)
}

/// Generate a valid similarity threshold.
fn arb_threshold() -> impl Strategy<Value = f64> {
    0.0f64..=1.0f64
}

/// Generate arbitrary integer values.
fn arb_int() -> impl Strategy<Value = i32> {
    any::<i32>()
}

/// Generate arbitrary string values.
fn arb_string() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9]{0,50}"
}

/// Generate a VectorPattern with optional threshold.
fn arb_vector_pattern(dims: usize) -> impl Strategy<Value = VectorPattern<f64>> {
    (arb_normalized_vector(dims), proptest::option::of(arb_threshold()))
        .prop_map(|(query, threshold)| {
            match threshold {
                Some(t) => VectorPattern::with_threshold(query, t),
                None => VectorPattern::query(query),
            }
        })
}

// =============================================================================
// ExactMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: ExactMatch is reflexive.
    ///
    /// ∀ x. ExactMatch.matches(x, x) == true
    ///
    /// Rocq: `exact_match_reflexive` in Match.v
    #[test]
    fn prop_exact_match_reflexive(x in arb_int()) {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        prop_assert!(
            matcher.matches(&x, &x),
            "ExactMatch should be reflexive: {} should match itself", x
        );
    }

    /// Property: ExactMatch is symmetric (follows from PartialEq).
    ///
    /// ∀ x, y. ExactMatch.matches(x, y) == ExactMatch.matches(y, x)
    ///
    /// Rocq: `exact_match_symmetric` in Match.v
    #[test]
    fn prop_exact_match_symmetric(x in arb_int(), y in arb_int()) {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        prop_assert_eq!(
            matcher.matches(&x, &y),
            matcher.matches(&y, &x),
            "ExactMatch should be symmetric"
        );
    }

    /// Property: ExactMatch is transitive when matching.
    ///
    /// ∀ x, y, z. matches(x, y) ∧ matches(y, z) → matches(x, z)
    ///
    /// Rocq: `exact_match_transitive` in Match.v
    #[test]
    fn prop_exact_match_transitive(x in arb_int(), y in arb_int(), z in arb_int()) {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        if matcher.matches(&x, &y) && matcher.matches(&y, &z) {
            prop_assert!(
                matcher.matches(&x, &z),
                "ExactMatch should be transitive: if {} == {} and {} == {}, then {} == {}",
                x, y, y, z, x, z
            );
        }
    }

    /// Property: ExactMatch for strings is reflexive.
    ///
    /// Rocq: `exact_match_reflexive_strings` in Match.v
    #[test]
    fn prop_exact_match_reflexive_strings(s in arb_string()) {
        let matcher: ExactMatch<String> = ExactMatch::new();
        prop_assert!(
            matcher.matches(&s, &s),
            "ExactMatch should be reflexive for strings"
        );
    }

    /// Property: ExactMatch correctly reports non-matching values.
    ///
    /// ∀ x, y. x ≠ y → ExactMatch.matches(x, y) == false
    ///
    /// Rocq: `exact_match_distinct` in Match.v
    #[test]
    fn prop_exact_match_distinct(x in 0i32..1000, y in 1001i32..2000) {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        prop_assert!(
            !matcher.matches(&x, &y),
            "ExactMatch should not match distinct values {} and {}", x, y
        );
    }
}

// =============================================================================
// WildcardMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: WildcardMatch always returns true.
    ///
    /// ∀ p, d. WildcardMatch.matches(p, d) == true
    ///
    /// Rocq: `wildcard_always_matches` in Match.v
    #[test]
    fn prop_wildcard_always_matches(p in arb_int(), d in arb_string()) {
        let matcher: WildcardMatch<i32, String> = WildcardMatch::new();
        prop_assert!(
            matcher.matches(&p, &d),
            "WildcardMatch should always return true"
        );
    }

    /// Property: WildcardMatch is independent of pattern value.
    ///
    /// ∀ p1, p2, d. WildcardMatch.matches(p1, d) == WildcardMatch.matches(p2, d)
    ///
    /// Rocq: `wildcard_pattern_independent` in Match.v
    #[test]
    fn prop_wildcard_pattern_independent(p1 in arb_int(), p2 in arb_int(), d in arb_int()) {
        let matcher: WildcardMatch<i32, i32> = WildcardMatch::new();
        prop_assert_eq!(
            matcher.matches(&p1, &d),
            matcher.matches(&p2, &d),
            "WildcardMatch should be independent of pattern"
        );
    }

    /// Property: WildcardMatch is independent of data value.
    ///
    /// ∀ p, d1, d2. WildcardMatch.matches(p, d1) == WildcardMatch.matches(p, d2)
    ///
    /// Rocq: `wildcard_data_independent` in Match.v
    #[test]
    fn prop_wildcard_data_independent(p in arb_int(), d1 in arb_int(), d2 in arb_int()) {
        let matcher: WildcardMatch<i32, i32> = WildcardMatch::new();
        prop_assert_eq!(
            matcher.matches(&p, &d1),
            matcher.matches(&p, &d2),
            "WildcardMatch should be independent of data"
        );
    }
}

// =============================================================================
// VectorDBMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Cosine similarity of a normalized vector with itself is 1.0.
    ///
    /// ∀ v normalized. cosine_similarity(v, v) ≈ 1.0
    ///
    /// Rocq: `cosine_similarity_identity` in Collections/VectorDB.v
    #[test]
    fn prop_cosine_similarity_identity(v in arb_normalized_vector(3..=100)) {
        let sim = VectorDBMatch::cosine_similarity(&v, &v);
        prop_assert!(
            (sim - 1.0).abs() < 0.0001,
            "Cosine similarity of normalized vector with itself should be 1.0, got {}",
            sim
        );
    }

    /// Property: Cosine similarity is bounded in [-1, 1].
    ///
    /// ∀ a, b. -1.0 ≤ cosine_similarity(a, b) ≤ 1.0
    ///
    /// Rocq: `cosine_similarity_bounded` in Collections/VectorDB.v
    #[test]
    fn prop_cosine_similarity_bounded(
        a in arb_f64_vector(3..=50),
        b in arb_f64_vector(3..=50)
    ) {
        // Only test when dimensions match
        if a.len() == b.len() && !a.is_empty() {
            let sim = VectorDBMatch::cosine_similarity(&a, &b);
            prop_assert!(
                sim >= -1.0 - 0.0001 && sim <= 1.0 + 0.0001,
                "Cosine similarity should be in [-1, 1], got {}", sim
            );
        }
    }

    /// Property: Cosine similarity is symmetric.
    ///
    /// ∀ a, b. cosine_similarity(a, b) == cosine_similarity(b, a)
    ///
    /// Rocq: `cosine_similarity_symmetric` in Collections/VectorDB.v
    #[test]
    fn prop_cosine_similarity_symmetric(
        a in arb_normalized_vector(3..=50),
        b in arb_normalized_vector(3..=50)
    ) {
        if a.len() == b.len() {
            let sim_ab = VectorDBMatch::cosine_similarity(&a, &b);
            let sim_ba = VectorDBMatch::cosine_similarity(&b, &a);
            prop_assert!(
                (sim_ab - sim_ba).abs() < 0.0001,
                "Cosine similarity should be symmetric: {} vs {}", sim_ab, sim_ba
            );
        }
    }

    /// Property: VectorDBMatch respects threshold semantics.
    ///
    /// matches(p, d) == (cosine_similarity(p, d) >= threshold)
    ///
    /// Rocq: `vectordb_threshold_semantics` in Match.v
    #[test]
    fn prop_vectordb_threshold_semantics(
        threshold in arb_threshold(),
        v1 in arb_normalized_vector(10),
        v2 in arb_normalized_vector(10)
    ) {
        let matcher = VectorDBMatch::new(threshold);
        let similarity = VectorDBMatch::cosine_similarity(&v1, &v2);
        let matches = matcher.matches(&v1, &v2);
        let expected = similarity >= threshold;

        prop_assert_eq!(
            matches,
            expected,
            "Match result {} should equal (similarity {} >= threshold {})",
            matches, similarity, threshold
        );
    }

    /// Property: VectorDBMatch normalized vector matches itself with any threshold < 1.0.
    ///
    /// ∀ v normalized, t < 1.0. VectorDBMatch(t).matches(v, v) == true
    ///
    /// Rocq: `vectordb_reflexive_below_one` in Match.v
    #[test]
    fn prop_vectordb_reflexive_below_one(
        v in arb_normalized_vector(3..=50),
        threshold in 0.0f64..0.99f64
    ) {
        let matcher = VectorDBMatch::new(threshold);
        prop_assert!(
            matcher.matches(&v, &v),
            "Normalized vector should match itself with threshold {}", threshold
        );
    }

    /// Property: Lower threshold accepts more matches.
    ///
    /// ∀ t1 < t2, p, d. VectorDBMatch(t2).matches(p, d) → VectorDBMatch(t1).matches(p, d)
    ///
    /// Rocq: `vectordb_threshold_monotonic` in Match.v
    #[test]
    fn prop_vectordb_threshold_monotonic(
        v1 in arb_normalized_vector(10),
        v2 in arb_normalized_vector(10),
        t1 in 0.0f64..0.5f64,
        t2 in 0.5f64..1.0f64
    ) {
        let matcher_high = VectorDBMatch::new(t2);
        let matcher_low = VectorDBMatch::new(t1);

        // If high threshold matches, low threshold must also match
        if matcher_high.matches(&v1, &v2) {
            prop_assert!(
                matcher_low.matches(&v1, &v2),
                "Lower threshold should accept at least as many matches"
            );
        }
    }

    /// Property: Empty vectors return 0 similarity.
    ///
    /// Rocq: `cosine_similarity_empty_zero` in Collections/VectorDB.v
    #[test]
    fn prop_cosine_similarity_empty_zero(_seed in any::<u64>()) {
        let empty: Vec<f64> = vec![];
        let v = vec![1.0, 2.0, 3.0];

        let sim1 = VectorDBMatch::cosine_similarity(&empty, &v);
        let sim2 = VectorDBMatch::cosine_similarity(&v, &empty);
        let sim3 = VectorDBMatch::cosine_similarity(&empty, &empty);

        prop_assert_eq!(sim1, 0.0, "Empty vector similarity should be 0");
        prop_assert_eq!(sim2, 0.0, "Empty vector similarity should be 0");
        prop_assert_eq!(sim3, 0.0, "Empty vector similarity should be 0");
    }

    /// Property: Mismatched dimensions return 0 similarity.
    ///
    /// Rocq: `cosine_similarity_dimension_mismatch` in Collections/VectorDB.v
    #[test]
    fn prop_cosine_similarity_dimension_mismatch(
        v1 in arb_f64_vector(3..=5),
        v2 in arb_f64_vector(6..=10)
    ) {
        // Dimensions are guaranteed different by generator ranges
        let sim = VectorDBMatch::cosine_similarity(&v1, &v2);
        prop_assert_eq!(
            sim,
            0.0,
            "Mismatched dimensions should return 0 similarity"
        );
    }

    /// Property: VectorDBMatch f32 vectors work correctly.
    ///
    /// Rocq: `vectordb_f32_consistent` in Match.v
    #[test]
    fn prop_vectordb_f32_consistent(
        threshold in arb_threshold(),
        v in arb_f32_vector(10)
    ) {
        let matcher = VectorDBMatch::new(threshold);
        // A vector should match itself if threshold <= 1.0
        let sim = VectorDBMatch::cosine_similarity_f32(&v, &v);
        let expected = sim >= threshold as f32;
        prop_assert_eq!(
            matcher.matches(&v, &v),
            expected,
            "f32 vector self-matching should follow threshold semantics"
        );
    }
}

// =============================================================================
// VectorPattern Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Per-pattern threshold overrides matcher default.
    ///
    /// ∀ pattern with threshold t, matcher with default d.
    ///   effective_threshold = t (not d)
    ///
    /// Rocq: `per_pattern_threshold_override` in Match.v
    #[test]
    fn prop_per_pattern_threshold_override(
        pattern_threshold in arb_threshold(),
        default_threshold in arb_threshold()
    ) {
        let query = vec![1.0, 0.0, 0.0];
        let pattern = VectorPattern::with_threshold(query, pattern_threshold);
        let effective = pattern.effective_threshold(default_threshold);

        prop_assert_eq!(
            effective,
            pattern_threshold,
            "Per-pattern threshold should override default"
        );
    }

    /// Property: Pattern without threshold uses matcher default.
    ///
    /// ∀ pattern without threshold, matcher with default d.
    ///   effective_threshold = d
    ///
    /// Rocq: `pattern_uses_default_threshold` in Match.v
    #[test]
    fn prop_pattern_uses_default_threshold(default_threshold in arb_threshold()) {
        let query = vec![1.0, 0.0, 0.0];
        let pattern = VectorPattern::query(query);
        let effective = pattern.effective_threshold(default_threshold);

        prop_assert_eq!(
            effective,
            default_threshold,
            "Pattern without threshold should use default"
        );
    }

    /// Property: Per-pattern threshold is clamped to [0, 1].
    ///
    /// Rocq: `pattern_threshold_clamped` in Match.v
    #[test]
    fn prop_pattern_threshold_clamped(raw_threshold in -10.0f64..10.0f64) {
        let query = vec![1.0, 0.0, 0.0];
        let pattern = VectorPattern::with_threshold(query, raw_threshold);
        let effective = pattern.threshold.unwrap();

        prop_assert!(
            effective >= 0.0 && effective <= 1.0,
            "Pattern threshold should be clamped to [0, 1], got {}",
            effective
        );
    }

    /// Property: VectorPattern matches correctly with per-pattern threshold.
    ///
    /// Rocq: `vector_pattern_matches_semantics` in Match.v
    #[test]
    fn prop_vector_pattern_matches_semantics(
        matcher_threshold in arb_threshold(),
        pattern_threshold in arb_threshold(),
        query in arb_normalized_vector(10),
        data in arb_normalized_vector(10)
    ) {
        let matcher = VectorDBMatch::new(matcher_threshold);
        let pattern = VectorPattern::with_threshold(query.clone(), pattern_threshold);

        let matches = matcher.matches(&pattern, &data);
        let similarity = VectorDBMatch::cosine_similarity(&query, &data);
        let expected = similarity >= pattern_threshold;

        prop_assert_eq!(
            matches,
            expected,
            "VectorPattern should use its own threshold, not matcher default"
        );
    }
}

// =============================================================================
// AndMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: AndMatch is conjunction of two matchers.
    ///
    /// ∀ m1, m2, p, d. AndMatch(m1, m2).matches(p, d) == m1.matches(p, d) ∧ m2.matches(p, d)
    ///
    /// Rocq: `and_match_conjunction` in Match.v
    #[test]
    fn prop_and_match_conjunction(x in arb_int(), y in arb_int()) {
        let exact1: ExactMatch<i32> = ExactMatch::new();
        let exact2: ExactMatch<i32> = ExactMatch::new();
        let and_matcher = AndMatch::new(ExactMatch::<i32>::new(), ExactMatch::<i32>::new());

        let result = and_matcher.matches(&x, &y);
        let expected = exact1.matches(&x, &y) && exact2.matches(&x, &y);

        prop_assert_eq!(
            result,
            expected,
            "AndMatch should be conjunction of component matchers"
        );
    }

    /// Property: AndMatch with wildcard degenerates to other matcher.
    ///
    /// ∀ m, p, d. AndMatch(m, wildcard).matches(p, d) == m.matches(p, d)
    ///
    /// Rocq: `and_match_wildcard_identity` in Match.v
    #[test]
    fn prop_and_match_wildcard_identity(x in arb_int(), y in arb_int()) {
        let exact: ExactMatch<i32> = ExactMatch::new();
        let wildcard: WildcardMatch<i32, i32> = WildcardMatch::new();
        let and_matcher = AndMatch::new(ExactMatch::<i32>::new(), wildcard);

        let result = and_matcher.matches(&x, &y);
        let expected = exact.matches(&x, &y);

        prop_assert_eq!(
            result,
            expected,
            "AndMatch with wildcard should equal other matcher"
        );
    }

    /// Property: AndMatch is commutative (order of matchers doesn't matter for result).
    ///
    /// ∀ m1, m2, p, d. AndMatch(m1, m2).matches(p, d) == AndMatch(m2, m1).matches(p, d)
    ///
    /// Rocq: `and_match_commutative` in Match.v
    #[test]
    fn prop_and_match_commutative(x in arb_int(), y in arb_int()) {
        let exact1: ExactMatch<i32> = ExactMatch::new();
        let exact2: ExactMatch<i32> = ExactMatch::new();
        let and_12 = AndMatch::new(exact1, exact2);
        let and_21 = AndMatch::new(ExactMatch::<i32>::new(), ExactMatch::<i32>::new());

        prop_assert_eq!(
            and_12.matches(&x, &y),
            and_21.matches(&x, &y),
            "AndMatch should be commutative"
        );
    }
}

// =============================================================================
// OrMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: OrMatch is disjunction of two matchers.
    ///
    /// ∀ m1, m2, p, d. OrMatch(m1, m2).matches(p, d) == m1.matches(p, d) ∨ m2.matches(p, d)
    ///
    /// Rocq: `or_match_disjunction` in Match.v
    #[test]
    fn prop_or_match_disjunction(x in arb_int(), y in arb_int()) {
        let exact1: ExactMatch<i32> = ExactMatch::new();
        let exact2: ExactMatch<i32> = ExactMatch::new();
        let or_matcher = OrMatch::new(ExactMatch::<i32>::new(), ExactMatch::<i32>::new());

        let result = or_matcher.matches(&x, &y);
        let expected = exact1.matches(&x, &y) || exact2.matches(&x, &y);

        prop_assert_eq!(
            result,
            expected,
            "OrMatch should be disjunction of component matchers"
        );
    }

    /// Property: OrMatch with wildcard always matches.
    ///
    /// ∀ m, p, d. OrMatch(m, wildcard).matches(p, d) == true
    ///
    /// Rocq: `or_match_wildcard_absorbs` in Match.v
    #[test]
    fn prop_or_match_wildcard_absorbs(x in arb_int(), y in arb_int()) {
        let exact: ExactMatch<i32> = ExactMatch::new();
        let wildcard: WildcardMatch<i32, i32> = WildcardMatch::new();
        let or_matcher = OrMatch::new(exact, wildcard);

        prop_assert!(
            or_matcher.matches(&x, &y),
            "OrMatch with wildcard should always match"
        );
    }

    /// Property: OrMatch is commutative.
    ///
    /// ∀ m1, m2, p, d. OrMatch(m1, m2).matches(p, d) == OrMatch(m2, m1).matches(p, d)
    ///
    /// Rocq: `or_match_commutative` in Match.v
    #[test]
    fn prop_or_match_commutative(x in arb_int(), y in arb_int()) {
        let exact1: ExactMatch<i32> = ExactMatch::new();
        let exact2: ExactMatch<i32> = ExactMatch::new();
        let or_12 = OrMatch::new(exact1, exact2);
        let or_21 = OrMatch::new(ExactMatch::<i32>::new(), ExactMatch::<i32>::new());

        prop_assert_eq!(
            or_12.matches(&x, &y),
            or_21.matches(&x, &y),
            "OrMatch should be commutative"
        );
    }
}

// =============================================================================
// PredicateMatcher Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: PredicateMatcher follows predicate semantics.
    ///
    /// ∀ p, d, pred. PredicateMatcher(pred).matches(p, d) == pred(p, d)
    ///
    /// Rocq: `predicate_matcher_semantics` in Match.v
    #[test]
    fn prop_predicate_matcher_semantics(p in arb_int(), d in arb_int()) {
        fn less_than(p: &i32, d: &i32) -> bool {
            *p < *d
        }

        let matcher = PredicateMatcher::new(
            less_than as fn(&i32, &i32) -> bool,
            "LessThan",
        );

        let result = matcher.matches(&p, &d);
        let expected = p < d;

        prop_assert_eq!(
            result,
            expected,
            "PredicateMatcher should follow predicate semantics"
        );
    }
}

// =============================================================================
// BoxedMatch Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Boxed matcher preserves behavior.
    ///
    /// ∀ m, p, d. boxed(m).matches(p, d) == m.matches(p, d)
    ///
    /// Rocq: `boxed_match_preserves_behavior` in Match.v
    #[test]
    fn prop_boxed_match_preserves_behavior(x in arb_int(), y in arb_int()) {
        let exact: ExactMatch<i32> = ExactMatch::new();
        let boxed_exact = boxed(ExactMatch::<i32>::new());

        prop_assert_eq!(
            exact.matches(&x, &y),
            boxed_exact.matches(&x, &y),
            "Boxed matcher should preserve behavior"
        );
    }
}

// =============================================================================
// Extract Bindings Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: extract_bindings returns Some iff matches returns true.
    ///
    /// ∀ m, p, d. m.extract_bindings(p, d).is_some() == m.matches(p, d)
    ///
    /// Rocq: `extract_bindings_consistent` in Match.v
    #[test]
    fn prop_extract_bindings_consistent(x in arb_int(), y in arb_int()) {
        let matcher: ExactMatch<i32> = ExactMatch::new();

        let matches = matcher.matches(&x, &y);
        let bindings = matcher.extract_bindings(&x, &y);

        prop_assert_eq!(
            bindings.is_some(),
            matches,
            "extract_bindings should return Some iff matches returns true"
        );
    }
}

// =============================================================================
// Static Invariant Tests (Non-Proptest)
// =============================================================================

#[test]
fn test_matcher_name_exact() {
    let matcher: ExactMatch<i32> = ExactMatch::new();
    assert_eq!(matcher.matcher_name(), "ExactMatch");
}

#[test]
fn test_matcher_name_wildcard() {
    let matcher: WildcardMatch<i32, i32> = WildcardMatch::new();
    assert_eq!(matcher.matcher_name(), "WildcardMatch");
}

#[test]
fn test_matcher_name_vectordb() {
    let matcher = VectorDBMatch::new(0.8);
    assert_eq!(
        <VectorDBMatch as Match<Vec<f64>, Vec<f64>>>::matcher_name(&matcher),
        "VectorDBMatch"
    );
}

#[test]
fn test_vectordb_default_threshold() {
    let matcher = VectorDBMatch::default_threshold();
    assert_eq!(matcher.threshold, 0.8);
}

#[test]
fn test_vectordb_threshold_clamp() {
    let matcher_low = VectorDBMatch::new(-1.0);
    assert_eq!(matcher_low.threshold, 0.0);

    let matcher_high = VectorDBMatch::new(2.0);
    assert_eq!(matcher_high.threshold, 1.0);
}

#[test]
fn test_vector_pattern_from_vec() {
    let v: Vec<f64> = vec![1.0, 2.0, 3.0];
    let pattern: VectorPattern<f64> = v.clone().into();
    assert_eq!(pattern.query, v);
    assert!(pattern.threshold.is_none());
}

#[test]
fn test_vector_pattern_from_tuple() {
    let v: Vec<f64> = vec![1.0, 2.0, 3.0];
    let pattern: VectorPattern<f64> = (v.clone(), 0.9).into();
    assert_eq!(pattern.query, v);
    assert_eq!(pattern.threshold, Some(0.9));
}

#[test]
fn test_cosine_distance() {
    let v = vec![1.0, 0.0, 0.0];
    let distance = VectorDBMatch::cosine_distance(&v, &v);
    assert!((distance - 0.0).abs() < 0.0001);
}

#[test]
fn test_and_match_name() {
    let exact1: ExactMatch<i32> = ExactMatch::new();
    let exact2: ExactMatch<i32> = ExactMatch::new();
    let and_matcher = AndMatch::new(exact1, exact2);
    assert_eq!(and_matcher.matcher_name(), "AndMatch");
}

#[test]
fn test_or_match_name() {
    let exact1: ExactMatch<i32> = ExactMatch::new();
    let exact2: ExactMatch<i32> = ExactMatch::new();
    let or_matcher = OrMatch::new(exact1, exact2);
    assert_eq!(or_matcher.matcher_name(), "OrMatch");
}
