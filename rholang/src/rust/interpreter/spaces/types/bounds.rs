//! Type Bound Trait Aliases
//!
//! This module defines trait aliases for the common type bounds used throughout
//! the multi-space RSpace integration. These aliases reduce boilerplate and
//! improve consistency.
//!
//! # Usage
//!
//! Instead of writing:
//! ```ignore
//! where
//!     C: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
//!     P: Clone + Send + Sync + 'static,
//!     A: Clone + Send + Sync + 'static,
//!     K: Clone + Send + Sync + 'static,
//! ```
//!
//! You can write:
//! ```ignore
//! where
//!     C: ChannelBound,
//!     P: PatternBound,
//!     A: DataBound,
//!     K: ContinuationBound,
//! ```
//!
//! # Note
//!
//! These are not true trait aliases (which require nightly Rust), but rather
//! empty traits with blanket implementations for all types satisfying the bounds.

use std::hash::Hash;

// ============================================================================
// Channel Type Bounds
// ============================================================================

/// Trait alias for channel type bounds.
///
/// Channels must be:
/// - `Clone`: For use in multiple contexts
/// - `Eq` + `Hash`: For use as HashMap keys
/// - `Send + Sync`: For concurrent access
/// - `'static`: For storage in trait objects
pub trait ChannelBound: Clone + Eq + Hash + Send + Sync + 'static {}

/// Blanket implementation for all types satisfying the bounds.
impl<T: Clone + Eq + Hash + Send + Sync + 'static> ChannelBound for T {}

// ============================================================================
// Pattern Type Bounds
// ============================================================================

/// Trait alias for pattern type bounds.
///
/// Patterns must be:
/// - `Clone`: For use in multiple contexts
/// - `Send + Sync`: For concurrent access
/// - `'static`: For storage in trait objects
pub trait PatternBound: Clone + Send + Sync + 'static {}

/// Blanket implementation for all types satisfying the bounds.
impl<T: Clone + Send + Sync + 'static> PatternBound for T {}

// ============================================================================
// Data Type Bounds
// ============================================================================

/// Trait alias for data type bounds.
///
/// Data must be:
/// - `Clone`: For use in multiple contexts
/// - `Send + Sync`: For concurrent access
/// - `'static`: For storage in trait objects
pub trait DataBound: Clone + Send + Sync + 'static {}

/// Blanket implementation for all types satisfying the bounds.
impl<T: Clone + Send + Sync + 'static> DataBound for T {}

// ============================================================================
// Continuation Type Bounds
// ============================================================================

/// Trait alias for continuation type bounds.
///
/// Continuations must be:
/// - `Clone`: For use in multiple contexts
/// - `Send + Sync`: For concurrent access
/// - `'static`: For storage in trait objects
pub trait ContinuationBound: Clone + Send + Sync + 'static {}

/// Blanket implementation for all types satisfying the bounds.
impl<T: Clone + Send + Sync + 'static> ContinuationBound for T {}

// ============================================================================
// Combined Bounds (for convenience)
// ============================================================================

/// Marker trait for types that satisfy all space parameter bounds.
///
/// This is a convenience trait for functions that need all four type parameters.
/// It doesn't add any new constraints, just combines the individual bounds.
pub trait SpaceParamBound: Clone + Send + Sync + 'static {}

/// Blanket implementation.
impl<T: Clone + Send + Sync + 'static> SpaceParamBound for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_channel_bound() {
        // String satisfies ChannelBound
        fn requires_channel_bound<C: ChannelBound>() {}
        requires_channel_bound::<String>();
        requires_channel_bound::<i32>();
        requires_channel_bound::<Vec<u8>>();
    }

    #[test]
    fn test_pattern_bound() {
        fn requires_pattern_bound<P: PatternBound>() {}
        requires_pattern_bound::<String>();
        requires_pattern_bound::<Vec<i32>>();
    }

    #[test]
    fn test_data_bound() {
        fn requires_data_bound<A: DataBound>() {}
        requires_data_bound::<String>();
        requires_data_bound::<Vec<u8>>();
    }

    #[test]
    fn test_continuation_bound() {
        fn requires_continuation_bound<K: ContinuationBound>() {}
        requires_continuation_bound::<String>();
        // Use Arc instead of Box since Arc<dyn Fn()> implements Clone
        requires_continuation_bound::<Arc<dyn Fn() + Send + Sync>>();
    }

    #[test]
    fn test_combined_usage() {
        // Test that we can use all bounds together in a where clause
        fn space_operation<C, P, A, K>()
        where
            C: ChannelBound,
            P: PatternBound,
            A: DataBound,
            K: ContinuationBound,
        {
            // This function compiles if the bounds work correctly
        }

        space_operation::<String, String, String, String>();
        // Use Arc instead of Box since Arc<dyn Fn()> implements Clone
        space_operation::<Vec<u8>, Vec<i32>, String, Arc<dyn Fn() + Send + Sync>>();
    }

    #[test]
    fn test_hashmap_with_channel_bound() {
        // ChannelBound types can be used as HashMap keys
        fn use_as_key<C: ChannelBound>() {
            let _map: HashMap<C, i32> = HashMap::new();
        }
        use_as_key::<String>();
    }
}
