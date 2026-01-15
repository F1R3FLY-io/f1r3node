//! Core Types for Multi-Space RSpace Integration
//!
//! This module defines the fundamental types for the 6-layer trait hierarchy
//! as specified in the "Reifying RSpaces" specification.
//!
//! # Module Organization
//!
//! - `theory`: Theory and Validatable traits for MeTTaIL integration
//! - `collections`: Inner collection types (Bag, Queue, Stack, Set, Cell, etc.)
//!   and outer storage types (HashMap, PathMap, Array, Vector, HashSet)
//! - `pathmap`: PathMap prefix aggregation and Par-to-Path encoding
//! - `qualifier`: Space qualifiers (Default, Temp, Seq)
//! - `id`: Space identification (SpaceId)
//! - `config`: Space configuration and gas configuration
//! - `raw_config`: Generic pass-through configuration for VectorDB backends

mod allocation;
mod bounds;
mod theory;
mod collections;
mod pathmap;
mod qualifier;
mod id;
mod config;
mod raw_config;

// ==========================================================================
// Re-exports for backward compatibility
// ==========================================================================

// From allocation module
pub use allocation::AllocationMode;

// From bounds module (type bound trait aliases)
pub use bounds::{
    ChannelBound,
    PatternBound,
    DataBound,
    ContinuationBound,
    SpaceParamBound,
};

// From theory module
pub use theory::{
    BoxedTheory,
    Theory,
    NullTheory,
    SimpleTypeTheory,
    Validatable,
    TheoryValidator,
    ValidationResult,
};

// From collections module
pub use collections::{
    InnerCollectionType,
    OuterStorageType,
    HyperparamSchema,
};

// From pathmap module
pub use pathmap::{
    // Aggregation types
    SuffixKey,
    AggregatedDatum,
    // Path prefix utilities
    get_path_suffix,
    path_prefixes,
    is_path_prefix,
    path_element_boundaries,
    // Varint encoding (for external use)
    encode_varint,
    decode_varint,
    // Par-to-Path conversion
    par_to_path,
    path_to_par,
    is_par_path,
    // Path tag constants
    path_tags,
};

// From qualifier module
pub use qualifier::SpaceQualifier;

// From id module
pub use id::SpaceId;

// From config module
pub use config::{
    GasConfiguration,
    SpaceConfig,
};

// From raw_config module (VectorDB pass-through configuration)
pub use raw_config::{
    RawConfigValue,
    RawVectorDBConfig,
    RawConfigError,
    par_to_raw_config_value,
    parse_raw_vectordb_config,
};
