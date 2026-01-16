//! Prelude module for common reified RSpaces imports.
//!
//! This module provides a curated set of the most commonly used types and traits
//! for working with reified RSpaces. Use `use rholang::spaces::prelude::*;` for
//! quick access to essential items.
//!
//! For less common items, import from specific submodules:
//! - `use rholang::spaces::agent::*;` - Core agent traits
//! - `use rholang::spaces::collections::*;` - Collection types
//! - `use rholang::spaces::matcher::*;` - Pattern matching
//! - `use rholang::spaces::factory::*;` - Space construction
//! - `use rholang::spaces::vectordb::*;` - Vector database integration

// Core types
pub use super::types::{
    SpaceId,
    SpaceQualifier,
    SpaceConfig,
    InnerCollectionType,
    OuterStorageType,
};

// Core traits
pub use super::agent::{SpaceAgent, CheckpointableSpace, ReplayableSpace};
pub use super::errors::SpaceError;

// Main implementation
pub use super::generic_rspace::{GenericRSpace, GenericRSpaceBuilder};

// Pattern matching
pub use super::matcher::{Match, ExactMatch};

// Checkpointing
pub use rspace_plus_plus::rspace::checkpoint::{Checkpoint, SoftCheckpoint};
pub use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

// Factory
pub use super::factory::{SpaceFactory, config_from_urn, urn_from_config};

// Registry
pub use super::registry::SpaceRegistry;
