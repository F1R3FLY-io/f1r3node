//! High-level VectorDB interface for application integration.
//!
//! This module defines the `VectorDB` trait, which composes a `VectorBackend`
//! with application-specific semantics like data association, persistence
//! flags, and consume-on-match behavior.

mod traits;
pub mod config;

pub use traits::VectorDB;
pub use config::{parse_config, RawConfigValue, VectorDBConfig};

// Feature-gated default implementation
mod default;

pub use default::{DefaultVectorDB, SimpleVectorDB, VectorDBBuilder};
