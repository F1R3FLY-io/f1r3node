//! Matcher-Based Extension Traits
//!
//! These extension traits provide matcher-aware methods for collections,
//! bridging the collection API with the Match<P, A> trait.

use super::super::matcher::Match;
use super::core::{DataCollection, ContinuationCollection};

/// Extension trait for DataCollection with matcher-based operations.
///
/// This trait provides matcher-aware versions of find and peek operations,
/// allowing collections to be used with the `Match<P, A>` trait.
pub trait DataCollectionExt<A>: DataCollection<A> {
    /// Find and remove data that matches the pattern using the given matcher.
    ///
    /// Returns `Some(data)` if a match was found and removed, `None` otherwise.
    fn find_and_remove_with_matcher<P>(
        &mut self,
        pattern: &P,
        matcher: &dyn Match<P, A>,
    ) -> Option<A>
    where
        A: Clone;

    /// Peek at data that matches the pattern without removing it.
    fn peek_with_matcher<P>(
        &self,
        pattern: &P,
        matcher: &dyn Match<P, A>,
    ) -> Option<&A>;
}

impl<A, T> DataCollectionExt<A> for T
where
    T: DataCollection<A>,
    A: Clone,
{
    fn find_and_remove_with_matcher<P>(
        &mut self,
        pattern: &P,
        matcher: &dyn Match<P, A>,
    ) -> Option<A>
    where
        A: Clone,
    {
        self.find_and_remove(|data| matcher.matches(pattern, data))
    }

    fn peek_with_matcher<P>(
        &self,
        pattern: &P,
        matcher: &dyn Match<P, A>,
    ) -> Option<&A> {
        self.peek(|data| matcher.matches(pattern, data))
    }
}

/// Extension trait for ContinuationCollection with matcher-based operations.
///
/// This trait provides methods to find continuations that would match given data.
pub trait ContinuationCollectionExt<P, K>: ContinuationCollection<P, K> {
    /// Find a continuation whose pattern matches the given data.
    ///
    /// Returns the continuation and its patterns if found.
    fn find_matching_for_data<A>(
        &mut self,
        data: &A,
        matcher: &dyn Match<P, A>,
    ) -> Option<(Vec<P>, K, bool)>
    where
        P: Clone,
        K: Clone;

    /// Peek at a continuation whose pattern matches the given data.
    fn peek_matching_for_data<A>(
        &self,
        data: &A,
        matcher: &dyn Match<P, A>,
    ) -> Option<(&[P], &K, bool)>;
}

impl<P, K, T> ContinuationCollectionExt<P, K> for T
where
    T: ContinuationCollection<P, K>,
    P: Clone,
    K: Clone,
{
    fn find_matching_for_data<A>(
        &mut self,
        data: &A,
        matcher: &dyn Match<P, A>,
    ) -> Option<(Vec<P>, K, bool)>
    where
        P: Clone,
        K: Clone,
    {
        self.find_and_remove(|patterns, _cont| {
            // Check if any pattern matches the data
            patterns.iter().any(|p| matcher.matches(p, data))
        })
    }

    fn peek_matching_for_data<A>(
        &self,
        data: &A,
        matcher: &dyn Match<P, A>,
    ) -> Option<(&[P], &K, bool)> {
        self.all_continuations()
            .into_iter()
            .find(|(patterns, _cont, _persist)| {
                patterns.iter().any(|p| matcher.matches(p, data))
            })
    }
}
