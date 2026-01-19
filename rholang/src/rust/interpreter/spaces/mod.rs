//! Spaces Module - Foundation Traits, Collections & Outer Storage
//!
//! This module provides the foundation for reified RSpaces.

pub mod collections;
pub mod channel_store;
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
    InnerCollectionType,
    OuterStorageType,
};
pub use errors::SpaceError;
pub use channel_store::ChannelStore;
