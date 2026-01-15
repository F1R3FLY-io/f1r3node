//! Backend abstraction layer for vector storage.
//!
//! This module defines the `VectorBackend` trait, a high-level, backend-agnostic
//! interface that any vector database can implement (Pinecone, Qdrant, Milvus,
//! in-memory, etc.).
//!
//! # Design Philosophy
//!
//! The trait is intentionally minimal and high-level. Implementation details
//! like tombstones, free lists, and SIMD masks are NOT exposed - those are
//! internal optimizations that specific backends can employ.
//!
//! # Trait Hierarchy
//!
//! - [`VectorBackend`]: Core backend trait for vector storage and retrieval
//! - [`HandlerBackend`]: Extension trait for backends with custom handler support

mod traits;

pub use traits::VectorBackend;

pub use traits::HandlerBackend;

// Feature-gated in-memory implementation
mod in_memory;

pub use in_memory::InMemoryBackend;
