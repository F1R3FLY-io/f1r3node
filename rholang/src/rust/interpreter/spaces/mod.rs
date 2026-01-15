//! Spaces Module - Foundation Traits & Collections Structure
//!
//! This module provides the foundation traits for reified RSpaces.

pub mod collections;
pub mod errors;
pub mod matcher;
pub mod types;
pub mod vectordb;

// Re-exports for convenience
pub use types::{
    ChannelBound,
    PatternBound,
    DataBound,
    ContinuationBound,
    SpaceParamBound,
    SpaceId,
};
pub use errors::SpaceError;
