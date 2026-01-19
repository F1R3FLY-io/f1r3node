//! VectorDB Module - Base Support
//!
//! This module provides the foundation for VectorDB-style similarity matching.

pub mod types;
pub mod registry;
pub mod in_memory;

pub use types::EmbeddingType;
pub use registry::{BackendRegistry, VectorBackendProvider};
