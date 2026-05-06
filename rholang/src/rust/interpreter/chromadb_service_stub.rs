use std::sync::Arc;

/// ChromaDB service implementation using enum dispatch for async compatibility
/// This avoids the dyn-compatibility issues with async trait methods.
/// This version is a stub for cases when Chroma is not needed.
pub enum ChromaDBService {
    /// NoOp implementation that returns empty results
    NoOp
}

/// Type alias for thread-safe ChromaDB service
pub type SharedChromaDBService = Arc<ChromaDBService>;

/// Create a NoOp OpenAI service
pub fn create_noop_chromadb_service() -> SharedChromaDBService {
    Arc::new(ChromaDBService::NoOp)
}

/// This is a stub version of the function that creates a real ChromaDB service
/// Instead, it just creates a noop
pub fn create_chromadb_service() -> SharedChromaDBService {
    tracing::info!("ChromaDB will not be provided because this node was compiled without support");
    create_noop_chromadb_service()
}