//! Collections Module - Foundation Traits
//!
//! This module provides foundation traits for inner collections.

pub mod semantics;
pub mod core;
pub mod similarity;
pub mod extensions;

pub use semantics::{SemanticEq, SemanticHash, TopKEntry};
pub use core::{DataCollection, ContinuationCollection};
pub use similarity::{SimilarityCollection, StoredSimilarityInfo, ContinuationId, SimilarityQueryMatrix};
pub use extensions::{DataCollectionExt, ContinuationCollectionExt};
