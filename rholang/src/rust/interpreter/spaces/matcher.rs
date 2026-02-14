//! Match Trait for Pattern Matching Abstraction
//!
//! This module defines the `Match<P, A>` trait that abstracts pattern matching
//! for data collections. Different matchers can implement different matching
//! strategies:
//!
//! - **RholangMatch**: Structural spatial matching for Rholang patterns
//! - **VectorDBMatch**: Similarity-based matching using vector embeddings
//! - **ExactMatch**: Simple equality-based matching
//!
//! This abstraction allows GenericRSpace to be parameterized by the matching
//! strategy, enabling different spaces to use different matching semantics.

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Pattern matching trait for data collections.
///
/// This trait abstracts the matching logic used by produce and consume operations
/// in RSpace. By implementing this trait, different matching strategies can be
/// plugged into the same GenericRSpace implementation.
///
/// # Type Parameters
/// - `P`: Pattern type (what we're matching against)
/// - `A`: Data type (what we're matching)
///
/// # Safety
/// Implementations must be thread-safe (Send + Sync) because matching may occur
/// concurrently during space operations.
pub trait Match<P, A>: Send + Sync + Debug {
    /// Check if a pattern matches data.
    ///
    /// # Arguments
    /// - `pattern`: The pattern to match against
    /// - `data`: The data to check
    ///
    /// # Returns
    /// `true` if the pattern matches the data, `false` otherwise.
    fn matches(&self, pattern: &P, data: &A) -> bool;

    /// Extract bindings from a successful match.
    ///
    /// When a pattern contains variables (e.g., `x` in `for (x <- ch)`),
    /// this method extracts the bindings for those variables.
    ///
    /// # Arguments
    /// - `pattern`: The pattern that matched
    /// - `data`: The data that was matched
    ///
    /// # Returns
    /// - `Some(bindings)` if the pattern matches and bindings were extracted
    /// - `None` if the pattern doesn't match
    ///
    /// The binding keys are variable names, values are the matched data fragments.
    fn extract_bindings(&self, pattern: &P, data: &A) -> Option<HashMap<String, A>>
    where
        A: Clone,
    {
        // Default implementation: if it matches, return empty bindings
        if self.matches(pattern, data) {
            Some(HashMap::new())
        } else {
            None
        }
    }

    /// Get the name of this matcher for debugging/logging.
    fn matcher_name(&self) -> &'static str;
}

// ==========================================================================
// ExactMatch - Simple Equality-Based Matching
// ==========================================================================

/// Simple equality-based matcher.
///
/// This matcher uses the `PartialEq` implementation of the types to determine
/// if a pattern matches data. Useful for testing and simple use cases.
#[derive(Debug, Clone, Default)]
pub struct ExactMatch<T> {
    _phantom: PhantomData<T>,
}

impl<T> ExactMatch<T> {
    /// Create a new ExactMatch.
    pub fn new() -> Self {
        ExactMatch {
            _phantom: PhantomData,
        }
    }
}

impl<T: PartialEq + Clone + Send + Sync + Debug> Match<T, T> for ExactMatch<T> {
    fn matches(&self, pattern: &T, data: &T) -> bool {
        pattern == data
    }

    fn matcher_name(&self) -> &'static str {
        "ExactMatch"
    }
}

// ==========================================================================
// PredicateMatcher - Closure-Based Matching
// ==========================================================================

/// Matcher that uses a predicate function.
///
/// This allows wrapping arbitrary matching logic in a Match implementation.
/// Useful for bridging legacy code or one-off matching needs.
#[derive(Debug)]
pub struct PredicateMatcher<P, A, F>
where
    F: Fn(&P, &A) -> bool + Send + Sync,
{
    predicate: F,
    name: &'static str,
    _phantom: PhantomData<(P, A)>,
}

impl<P, A, F> PredicateMatcher<P, A, F>
where
    F: Fn(&P, &A) -> bool + Send + Sync,
{
    /// Create a new PredicateMatcher with the given predicate function.
    pub fn new(predicate: F, name: &'static str) -> Self {
        PredicateMatcher {
            predicate,
            name,
            _phantom: PhantomData,
        }
    }
}

impl<P, A, F> Match<P, A> for PredicateMatcher<P, A, F>
where
    P: Send + Sync + Debug,
    A: Send + Sync + Debug,
    F: Fn(&P, &A) -> bool + Send + Sync + Debug,
{
    fn matches(&self, pattern: &P, data: &A) -> bool {
        (self.predicate)(pattern, data)
    }

    fn matcher_name(&self) -> &'static str {
        self.name
    }
}

// ==========================================================================
// VectorDBMatch - Similarity-Based Matching (ndarray-accelerated)
// ==========================================================================

use ndarray::Array1;
// Vector operations from local tensor module
#[cfg(feature = "vectordb")]
use super::super::tensor as vector_ops;

/// Pattern for VectorDB matching with per-pattern threshold.
///
/// This allows each consume pattern to specify its own similarity bound,
/// as required by the Reifying RSpaces design document.
///
/// # Fields
/// - `query`: The query embedding vector
/// - `threshold`: Optional per-pattern threshold (uses matcher default if None)
#[derive(Debug, Clone)]
pub struct VectorPattern<T> {
    /// Query embedding vector
    pub query: Vec<T>,
    /// Per-pattern similarity threshold (None uses matcher default)
    pub threshold: Option<f64>,
}

impl<T> VectorPattern<T> {
    /// Create a new pattern with just a query vector (uses matcher default threshold).
    pub fn query(query: Vec<T>) -> Self {
        VectorPattern {
            query,
            threshold: None,
        }
    }

    /// Create a new pattern with a query vector and specific threshold.
    pub fn with_threshold(query: Vec<T>, threshold: f64) -> Self {
        VectorPattern {
            query,
            threshold: Some(threshold.clamp(0.0, 1.0)),
        }
    }

    /// Get the effective threshold for this pattern.
    pub fn effective_threshold(&self, default: f64) -> f64 {
        self.threshold.unwrap_or(default)
    }
}

impl<T: Clone> From<Vec<T>> for VectorPattern<T> {
    fn from(query: Vec<T>) -> Self {
        VectorPattern::query(query)
    }
}

impl<T: Clone> From<(Vec<T>, f64)> for VectorPattern<T> {
    fn from((query, threshold): (Vec<T>, f64)) -> Self {
        VectorPattern::with_threshold(query, threshold)
    }
}

/// Vector similarity-based matcher for embedding vectors.
///
/// Uses ndarray-accelerated cosine similarity to determine if a query vector
/// matches a data vector. A match occurs when the cosine similarity exceeds
/// the threshold (either per-pattern or the matcher default).
///
/// This is designed for AI/ML integration where semantic similarity is used
/// for matching rather than structural equality.
///
/// # Tensor Logic Integration
///
/// The threshold can be interpreted as a temperature parameter:
/// - threshold → 1.0: Deductive reasoning (exact matches only)
/// - threshold → 0.0: Analogical reasoning (any similarity)
#[derive(Debug, Clone)]
pub struct VectorDBMatch {
    /// Default minimum cosine similarity for a match (0.0 to 1.0)
    pub threshold: f64,
}

impl VectorDBMatch {
    /// Create a new VectorDBMatch with the given similarity threshold.
    ///
    /// # Arguments
    /// - `threshold`: Minimum cosine similarity for a match (clamped to 0.0..1.0)
    pub fn new(threshold: f64) -> Self {
        VectorDBMatch {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create a VectorDBMatch with the default threshold of 0.8.
    pub fn default_threshold() -> Self {
        Self::new(0.8)
    }

    /// Compute cosine similarity between two f64 vectors using ndarray.
    ///
    /// Uses BLAS-accelerated dot product for efficiency.
    ///
    /// # Returns
    /// Cosine similarity in range [-1.0, 1.0], or 0.0 if vectors are invalid.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        // Convert to ndarray for BLAS acceleration
        let arr_a = Array1::from_vec(a.iter().map(|&x| x as f32).collect());
        let arr_b = Array1::from_vec(b.iter().map(|&x| x as f32).collect());

        vector_ops::cosine_similarity_safe(&arr_a, &arr_b) as f64
    }

    /// Compute cosine similarity between two f32 vectors using ndarray.
    ///
    /// Uses BLAS-accelerated dot product for efficiency.
    ///
    /// # Returns
    /// Cosine similarity in range [-1.0, 1.0], or 0.0 if vectors are invalid.
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        // Convert to ndarray for BLAS acceleration
        let arr_a = Array1::from_vec(a.to_vec());
        let arr_b = Array1::from_vec(b.to_vec());

        vector_ops::cosine_similarity_safe(&arr_a, &arr_b)
    }

    /// Compute cosine distance (1 - similarity).
    pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        1.0 - Self::cosine_similarity(a, b)
    }

    /// Check if vectors match with a specific threshold.
    pub fn matches_with_threshold(pattern: &[f64], data: &[f64], threshold: f64) -> bool {
        Self::cosine_similarity(pattern, data) >= threshold
    }

    /// Check if f32 vectors match with a specific threshold.
    pub fn matches_with_threshold_f32(pattern: &[f32], data: &[f32], threshold: f32) -> bool {
        Self::cosine_similarity_f32(pattern, data) >= threshold
    }
}

impl Default for VectorDBMatch {
    fn default() -> Self {
        Self::default_threshold()
    }
}

// Support Vec<f64> pattern and data (backward compatible)
impl Match<Vec<f64>, Vec<f64>> for VectorDBMatch {
    fn matches(&self, pattern: &Vec<f64>, data: &Vec<f64>) -> bool {
        Self::cosine_similarity(pattern, data) >= self.threshold
    }

    fn matcher_name(&self) -> &'static str {
        "VectorDBMatch"
    }
}

// Support Vec<f32> pattern and data (backward compatible)
impl Match<Vec<f32>, Vec<f32>> for VectorDBMatch {
    fn matches(&self, pattern: &Vec<f32>, data: &Vec<f32>) -> bool {
        Self::cosine_similarity_f32(pattern, data) >= self.threshold as f32
    }

    fn matcher_name(&self) -> &'static str {
        "VectorDBMatch"
    }
}

// Support VectorPattern<f64> with per-pattern threshold
impl Match<VectorPattern<f64>, Vec<f64>> for VectorDBMatch {
    fn matches(&self, pattern: &VectorPattern<f64>, data: &Vec<f64>) -> bool {
        let effective_threshold = pattern.effective_threshold(self.threshold);
        Self::matches_with_threshold(&pattern.query, data, effective_threshold)
    }

    fn matcher_name(&self) -> &'static str {
        "VectorDBMatch(per-pattern)"
    }
}

// Support VectorPattern<f32> with per-pattern threshold
impl Match<VectorPattern<f32>, Vec<f32>> for VectorDBMatch {
    fn matches(&self, pattern: &VectorPattern<f32>, data: &Vec<f32>) -> bool {
        let effective_threshold = pattern.effective_threshold(self.threshold);
        Self::matches_with_threshold_f32(&pattern.query, data, effective_threshold as f32)
    }

    fn matcher_name(&self) -> &'static str {
        "VectorDBMatch(per-pattern)"
    }
}

// ==========================================================================
// WildcardMatch - Always Matches
// ==========================================================================

/// A matcher that always returns true.
///
/// Useful for "match any" patterns or as a placeholder in testing.
#[derive(Debug, Clone, Default)]
pub struct WildcardMatch<P, A> {
    _phantom: PhantomData<(P, A)>,
}

impl<P, A> WildcardMatch<P, A> {
    pub fn new() -> Self {
        WildcardMatch {
            _phantom: PhantomData,
        }
    }
}

impl<P: Send + Sync + Debug, A: Send + Sync + Debug> Match<P, A> for WildcardMatch<P, A> {
    fn matches(&self, _pattern: &P, _data: &A) -> bool {
        true
    }

    fn matcher_name(&self) -> &'static str {
        "WildcardMatch"
    }
}

// ==========================================================================
// ComposedMatch - Combine Multiple Matchers
// ==========================================================================

/// A matcher that combines two matchers with AND semantics.
///
/// Both matchers must match for the composed matcher to match.
#[derive(Debug)]
pub struct AndMatch<M1, M2, P, A>
where
    M1: Match<P, A>,
    M2: Match<P, A>,
{
    first: M1,
    second: M2,
    _phantom: PhantomData<(P, A)>,
}

impl<M1, M2, P, A> AndMatch<M1, M2, P, A>
where
    M1: Match<P, A>,
    M2: Match<P, A>,
{
    pub fn new(first: M1, second: M2) -> Self {
        AndMatch {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<M1, M2, P, A> Match<P, A> for AndMatch<M1, M2, P, A>
where
    M1: Match<P, A> + Debug,
    M2: Match<P, A> + Debug,
    P: Send + Sync + Debug,
    A: Send + Sync + Debug,
{
    fn matches(&self, pattern: &P, data: &A) -> bool {
        self.first.matches(pattern, data) && self.second.matches(pattern, data)
    }

    fn matcher_name(&self) -> &'static str {
        "AndMatch"
    }
}

/// A matcher that combines two matchers with OR semantics.
///
/// Either matcher matching is sufficient for the composed matcher to match.
#[derive(Debug)]
pub struct OrMatch<M1, M2, P, A>
where
    M1: Match<P, A>,
    M2: Match<P, A>,
{
    first: M1,
    second: M2,
    _phantom: PhantomData<(P, A)>,
}

impl<M1, M2, P, A> OrMatch<M1, M2, P, A>
where
    M1: Match<P, A>,
    M2: Match<P, A>,
{
    pub fn new(first: M1, second: M2) -> Self {
        OrMatch {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<M1, M2, P, A> Match<P, A> for OrMatch<M1, M2, P, A>
where
    M1: Match<P, A> + Debug,
    M2: Match<P, A> + Debug,
    P: Send + Sync + Debug,
    A: Send + Sync + Debug,
{
    fn matches(&self, pattern: &P, data: &A) -> bool {
        self.first.matches(pattern, data) || self.second.matches(pattern, data)
    }

    fn matcher_name(&self) -> &'static str {
        "OrMatch"
    }
}

// ==========================================================================
// BoxedMatch - Type-Erased Matcher
// ==========================================================================

/// Type-erased matcher for dynamic dispatch.
///
/// Use this when you need to store matchers of different concrete types
/// in the same collection or when the matcher type is determined at runtime.
pub type BoxedMatch<P, A> = Box<dyn Match<P, A>>;

/// Create a boxed matcher from any Match implementation.
pub fn boxed<P, A, M>(matcher: M) -> BoxedMatch<P, A>
where
    M: Match<P, A> + 'static,
{
    Box::new(matcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_equal() {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        assert!(matcher.matches(&42, &42));
    }

    #[test]
    fn test_exact_match_not_equal() {
        let matcher: ExactMatch<i32> = ExactMatch::new();
        assert!(!matcher.matches(&42, &43));
    }

    #[test]
    fn test_exact_match_strings() {
        let matcher: ExactMatch<String> = ExactMatch::new();
        assert!(matcher.matches(&"hello".to_string(), &"hello".to_string()));
        assert!(!matcher.matches(&"hello".to_string(), &"world".to_string()));
    }

    #[test]
    fn test_wildcard_match_always_matches() {
        let matcher: WildcardMatch<i32, String> = WildcardMatch::new();
        assert!(matcher.matches(&42, &"anything".to_string()));
        assert!(matcher.matches(&0, &"".to_string()));
    }

    #[test]
    fn test_vectordb_match_identical_vectors() {
        let matcher = VectorDBMatch::new(0.9);
        let v = vec![1.0, 0.0, 0.0];
        assert!(matcher.matches(&v, &v));
    }

    #[test]
    fn test_vectordb_match_orthogonal_vectors() {
        let matcher = VectorDBMatch::new(0.5);
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        // Cosine similarity of orthogonal vectors is 0
        assert!(!matcher.matches(&v1, &v2));
    }

    #[test]
    fn test_vectordb_match_similar_vectors() {
        let matcher = VectorDBMatch::new(0.9);
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.95, 0.31, 0.0]; // ~0.95 similarity
        assert!(matcher.matches(&v1, &v2));
    }

    #[test]
    fn test_vectordb_match_below_threshold() {
        let matcher = VectorDBMatch::new(0.99);
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.95, 0.31, 0.0]; // ~0.95 similarity, below 0.99 threshold
        assert!(!matcher.matches(&v1, &v2));
    }

    #[test]
    fn test_vectordb_cosine_similarity() {
        // Same direction, different magnitude -> similarity 1.0
        let v1 = vec![1.0, 1.0];
        let v2 = vec![2.0, 2.0];
        let sim = VectorDBMatch::cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 0.001);

        // Opposite direction -> similarity -1.0
        let v3 = vec![-1.0, -1.0];
        let sim2 = VectorDBMatch::cosine_similarity(&v1, &v3);
        assert!((sim2 - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_and_match() {
        let matcher1: ExactMatch<i32> = ExactMatch::new();
        let matcher2: ExactMatch<i32> = ExactMatch::new();
        let and_matcher = AndMatch::new(matcher1, matcher2);

        // Both must match
        assert!(and_matcher.matches(&42, &42));
        assert!(!and_matcher.matches(&42, &43));
    }

    #[test]
    fn test_or_match() {
        let exact: ExactMatch<i32> = ExactMatch::new();
        let wildcard: WildcardMatch<i32, i32> = WildcardMatch::new();
        let or_matcher = OrMatch::new(exact, wildcard);

        // Wildcard always matches, so OR always matches
        assert!(or_matcher.matches(&42, &42));
        assert!(or_matcher.matches(&42, &43));
    }

    #[test]
    fn test_predicate_matcher() {
        // Use a function pointer instead of closure since fn ptrs implement Debug
        fn greater_than(p: &i32, d: &i32) -> bool {
            d > p
        }
        let matcher = PredicateMatcher::new(
            greater_than as fn(&i32, &i32) -> bool,
            "GreaterThan",
        );

        assert!(matcher.matches(&10, &20));
        assert!(!matcher.matches(&20, &10));
    }

    #[test]
    fn test_boxed_match() {
        let matcher: BoxedMatch<i32, i32> = boxed(ExactMatch::new());
        assert!(matcher.matches(&42, &42));
        assert!(!matcher.matches(&42, &43));
    }

    #[test]
    fn test_extract_bindings_default() {
        let matcher: ExactMatch<i32> = ExactMatch::new();

        // When pattern matches, default returns empty bindings
        let bindings = matcher.extract_bindings(&42, &42);
        assert!(bindings.is_some());
        assert!(bindings.unwrap().is_empty());

        // When pattern doesn't match, returns None
        let bindings = matcher.extract_bindings(&42, &43);
        assert!(bindings.is_none());
    }

    #[test]
    fn test_matcher_names() {
        let exact: ExactMatch<i32> = ExactMatch::new();
        assert_eq!(exact.matcher_name(), "ExactMatch");

        let wildcard: WildcardMatch<i32, i32> = WildcardMatch::new();
        assert_eq!(wildcard.matcher_name(), "WildcardMatch");

        let vectordb = VectorDBMatch::new(0.8);
        assert_eq!(<VectorDBMatch as Match<Vec<f64>, Vec<f64>>>::matcher_name(&vectordb), "VectorDBMatch");
    }

    #[test]
    fn test_f32_vector_match() {
        let matcher = VectorDBMatch::new(0.9);
        let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
        let v2: Vec<f32> = vec![1.0, 0.0, 0.0];
        assert!(matcher.matches(&v1, &v2));
    }
}
