//! Multi-Space RSpace Integration Module
//!
//! This module implements the 6-layer trait hierarchy for reified RSpaces.

pub mod adapter;
pub mod agent;
pub mod async_agent;
pub mod channel_store;
pub mod charging_agent;
pub mod collections;
pub mod errors;
pub mod generic_rspace;
pub mod history;
pub mod matcher;
pub mod phlogiston;
pub mod prelude;
pub mod registry;
pub mod similarity_extraction;
pub mod types;
pub mod vectordb;

// Re-exports for convenience
pub use types::{
    InnerCollectionType,
    OuterStorageType,
    SpaceConfig,
    SpaceId,
    SpaceQualifier,
    ChannelBound,
    PatternBound,
    DataBound,
    ContinuationBound,
    SpaceParamBound,
    Theory,
    NullTheory,
    GasConfiguration,
};
pub use errors::SpaceError;
pub use agent::SpaceAgent;
pub use async_agent::AsyncSpaceAgent;
pub use channel_store::ChannelStore;
pub use generic_rspace::GenericRSpace;
pub use registry::SpaceRegistry;
