//! Backend Registry for VectorDB Factory Pattern
//!
//! This module defines the interface that VectorDB backends implement and the registry
//! that manages them. rholang owns these trait definitions - backend implementations
//! depend on rholang for these traits and implement them.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  rholang (DEFINES the interface)                            │
//! │  ├── VectorBackendFactory trait                             │
//! │  ├── BackendConfig struct                                   │
//! │  ├── VectorBackendDyn trait                                 │
//! │  ├── BackendRegistry                                        │
//! │  └── ResolvedArg enum                                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  rho-vectordb (IMPLEMENTS the interface)                    │
//! │  ├── RhoBackendFactory                                      │
//! │  ├── InMemoryBackendWrapper                                 │
//! │  └── register_with_rholang() function                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  future backends (IMPLEMENT the interface)                  │
//! │  └── PineconeBackendFactory, QdrantBackendFactory, etc.     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Backends register themselves with the registry:
//!
//! ```ignore
//! use rholang::interpreter::spaces::BackendRegistry;
//!
//! // Create empty registry
//! let mut registry = BackendRegistry::new();
//!
//! // Backend crates register themselves
//! rho_vectordb::register_with_rholang(&mut registry);
//!
//! // Create a backend by name
//! let backend = registry.create("rho", 384, &BackendConfig::default())?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use super::super::errors::SpaceError;

// ==========================================================================
// Resolved Argument Type
// ==========================================================================

/// Resolved argument for query functions.
///
/// Used to pass parameters to similarity/ranking functions through the
/// generic `VectorBackendDyn::query()` interface.
#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedArg {
    /// Integer argument (e.g., k for top-k)
    Int(i64),
    /// Floating point argument (e.g., threshold)
    Float(f64),
    /// String argument
    String(String),
    /// Boolean argument
    Bool(bool),
}

impl ResolvedArg {
    /// Try to extract as i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ResolvedArg::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ResolvedArg::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as &str
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ResolvedArg::String(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ResolvedArg::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ==========================================================================
// Backend Configuration
// ==========================================================================

/// Configuration for backend creation.
///
/// Populated from the Rholang VectorDBFactory configuration map.
/// Backend implementations extract the parameters they need.
///
/// # Example
///
/// ```ignore
/// // From Rholang:
/// // VectorDBFactory!({"backend": "pinecone", "api_key": "pk-xxx", "endpoint": "..."}, *space)
///
/// let config = BackendConfig {
///     api_key: Some("pk-xxx".to_string()),
///     endpoint: Some("https://my-index.pinecone.io".to_string()),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Default)]
pub struct BackendConfig {
    /// API key for cloud backends (Pinecone, Qdrant, etc.)
    pub api_key: Option<String>,

    /// Endpoint URL for cloud backends
    pub endpoint: Option<String>,

    /// Additional key-value options from the config map.
    /// Backend implementations can extract custom parameters from here.
    pub options: HashMap<String, String>,
}

// ==========================================================================
// VectorBackendFactory Trait
// ==========================================================================

/// Factory trait for creating VectorBackend instances.
///
/// Backends (rho-vectordb, pinecone, qdrant, etc.) implement this trait.
/// rholang owns the trait definition; backends depend on rholang for it.
///
/// # Implementation Notes
///
/// - `name()` returns the canonical name for this backend (e.g., "rho", "pinecone")
/// - `aliases()` returns alternative names (e.g., ["default", "memory"] for the "rho" backend)
/// - `create()` constructs a new backend instance with the given dimensions and config
///
/// # Example
///
/// ```ignore
/// pub struct MyBackendFactory;
///
/// impl VectorBackendFactory for MyBackendFactory {
///     fn name(&self) -> &str { "mybackend" }
///     fn aliases(&self) -> &[&str] { &["custom"] }
///     fn create(&self, dimensions: usize, config: &BackendConfig)
///         -> Result<Box<dyn VectorBackendDyn>, SpaceError>
///     {
///         Ok(Box::new(MyBackend::new(dimensions, config)?))
///     }
/// }
/// ```
pub trait VectorBackendFactory: Send + Sync {
    /// The unique name for this backend (e.g., "rho", "pinecone").
    ///
    /// This is the canonical identifier used in Rholang:
    /// `VectorDBFactory!({"backend": "rho", ...}, *space)`
    fn name(&self) -> &str;

    /// Alternative names/aliases (e.g., ["default", "memory", "inmemory"]).
    ///
    /// Aliases allow users to use convenient shorthand names.
    /// For example, "default" could map to the "rho" backend.
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Create a new backend instance.
    ///
    /// # Arguments
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `config` - Configuration parameters from the Rholang factory call
    ///
    /// # Returns
    /// A boxed trait object implementing `VectorBackendDyn`
    fn create(
        &self,
        dimensions: usize,
        config: &BackendConfig,
    ) -> Result<Box<dyn VectorBackendDyn>, SpaceError>;
}

// ==========================================================================
// VectorBackendDyn Trait
// ==========================================================================

/// Object-safe version of VectorBackend for dynamic dispatch.
///
/// This trait allows backends to be used through dynamic dispatch
/// without rholang needing to know the concrete backend type.
///
/// Backends implement this trait to provide the actual storage operations.
/// The methods mirror the essential operations needed for VectorDB spaces.
///
/// # Query Interface
///
/// The `query()` method uses string identifiers for similarity and ranking
/// functions (e.g., "cosine", "euclidean", "top_k", "all"). This allows
/// rholang to remain agnostic about which functions a backend supports -
/// each backend defines its own supported functions.
pub trait VectorBackendDyn: Send + Sync {
    /// Store an embedding and return its ID.
    ///
    /// # Arguments
    /// * `embedding` - The embedding vector to store
    ///
    /// # Returns
    /// The ID assigned to the stored embedding
    fn store(&mut self, embedding: &[f32]) -> Result<usize, SpaceError>;

    /// Retrieve an embedding by ID.
    ///
    /// # Arguments
    /// * `id` - The ID of the embedding to retrieve
    ///
    /// # Returns
    /// The embedding vector, or None if not found
    fn get(&self, id: usize) -> Option<Vec<f32>>;

    /// Remove an embedding by ID.
    ///
    /// # Arguments
    /// * `id` - The ID of the embedding to remove
    ///
    /// # Returns
    /// true if the embedding was removed, false if not found
    fn remove(&mut self, id: usize) -> bool;

    /// Get the dimensionality of embeddings in this backend.
    fn dimensions(&self) -> usize;

    /// Get the number of embeddings stored.
    fn len(&self) -> usize;

    /// Check if the backend is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all embeddings.
    fn clear(&mut self);

    /// Execute a similarity query with optional function overrides.
    ///
    /// Rholang passes function identifiers as strings. The backend
    /// interprets these and performs similarity/ranking computation.
    ///
    /// # Arguments
    /// * `embedding` - Query embedding vector
    /// * `similarity_fn` - Optional similarity function override (e.g., "cosine", "euclidean")
    /// * `threshold` - Optional threshold override (uses backend default if None)
    /// * `ranking_fn` - Optional ranking function override (e.g., "top_k", "all")
    /// * `params` - Additional parameters for similarity/ranking functions
    ///
    /// # Returns
    /// Vector of (index, score) pairs sorted by score descending
    fn query(
        &self,
        embedding: &[f32],
        similarity_fn: Option<&str>,
        threshold: Option<f32>,
        ranking_fn: Option<&str>,
        params: &[ResolvedArg],
    ) -> Result<Vec<(usize, f32)>, SpaceError>;

    /// Get the default similarity threshold for this backend.
    fn default_threshold(&self) -> f32 {
        0.8
    }

    /// Get the default similarity metric ID for this backend.
    fn default_similarity_fn(&self) -> &str {
        "cosine"
    }

    /// Get the default ranking function ID for this backend.
    fn default_ranking_fn(&self) -> &str {
        "all"
    }

    /// Get the list of supported similarity function IDs.
    fn supported_similarity_fns(&self) -> Vec<String>;

    /// Get the list of supported ranking function IDs.
    fn supported_ranking_fns(&self) -> Vec<String>;

    /// Create a cloned instance of this backend.
    ///
    /// This method is used to support cloning `VectorDBDataCollection`.
    /// Implementations should return a new backend with the same configuration
    /// and a copy of all stored embeddings.
    ///
    /// # Returns
    /// A boxed clone of this backend with all embeddings copied
    fn clone_boxed(&self) -> Box<dyn VectorBackendDyn>;

    /// Get all embedding IDs currently stored.
    ///
    /// This is used for cloning to iterate over all embeddings.
    ///
    /// # Returns
    /// A vector of all valid embedding IDs
    fn all_ids(&self) -> Vec<usize>;
}

// ==========================================================================
// Backend Registry
// ==========================================================================

/// Registry for VectorBackend factories.
///
/// Maps backend names (case-insensitive) to factory implementations.
/// rholang owns this registry and backends register themselves with it.
///
/// # Default Registry
///
/// Using `BackendRegistry::default()` or `BackendRegistry::with_defaults()` creates
/// a registry with the built-in in-memory backend pre-registered. This provides
/// immediate access to VectorDB functionality without requiring manual registration.
///
/// Use `BackendRegistry::new()` for an empty registry when you need explicit control
/// over which backends are available.
///
/// # Example
///
/// ```ignore
/// // Option 1: Registry with defaults (recommended for most use cases)
/// let registry = BackendRegistry::default();
/// assert!(registry.contains("rho"));
/// assert!(registry.contains("default")); // alias
///
/// // Option 2: Empty registry for explicit control
/// let mut registry = BackendRegistry::new();
/// // Manually register backends as needed
///
/// // Create a backend
/// let backend = registry.create("rho", 384, &BackendConfig::default())?;
/// ```
#[derive(Clone)]
pub struct BackendRegistry {
    factories: HashMap<String, Arc<dyn VectorBackendFactory>>,
}

impl BackendRegistry {
    /// Create an empty registry.
    ///
    /// Use this when you want explicit control over which backends are registered.
    /// For most use cases, prefer `BackendRegistry::default()` or `with_defaults()`
    /// which come pre-populated with the built-in in-memory backend.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Create a registry with built-in backends pre-registered.
    ///
    /// This registers the following backends:
    /// - `"rho"` (canonical name) - In-memory SIMD-optimized backend
    /// - `"default"`, `"memory"`, `"inmemory"` (aliases)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = BackendRegistry::with_defaults();
    /// assert!(registry.contains("rho"));
    /// assert!(registry.contains("default"));
    /// assert!(registry.contains("memory"));
    ///
    /// let backend = registry.create("rho", 384, &BackendConfig::default())?;
    /// ```
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        super::in_memory::register_with_rholang(&mut registry);
        registry
    }

    /// Register a backend factory.
    ///
    /// The factory's name and all aliases are registered (case-insensitive).
    ///
    /// # Arguments
    /// * `factory` - The factory to register
    pub fn register(&mut self, factory: Arc<dyn VectorBackendFactory>) {
        let name = factory.name().to_lowercase();
        self.factories.insert(name.clone(), Arc::clone(&factory));

        for alias in factory.aliases() {
            self.factories
                .insert(alias.to_lowercase(), Arc::clone(&factory));
        }
    }

    /// Look up a factory by name (case-insensitive).
    ///
    /// # Arguments
    /// * `name` - The name or alias of the backend
    ///
    /// # Returns
    /// The factory if found, or None
    pub fn get(&self, name: &str) -> Option<Arc<dyn VectorBackendFactory>> {
        self.factories.get(&name.to_lowercase()).cloned()
    }

    /// Check if a backend is registered.
    ///
    /// # Arguments
    /// * `name` - The name or alias to check
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(&name.to_lowercase())
    }

    /// Create a backend by name.
    ///
    /// # Arguments
    /// * `name` - The name or alias of the backend
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `config` - Configuration parameters
    ///
    /// # Returns
    /// A boxed backend instance, or an error if the backend is not found
    ///
    /// # Errors
    ///
    /// Returns `SpaceError::InvalidConfiguration` if the backend name is not registered.
    pub fn create(
        &self,
        name: &str,
        dimensions: usize,
        config: &BackendConfig,
    ) -> Result<Box<dyn VectorBackendDyn>, SpaceError> {
        let factory = self.get(name).ok_or_else(|| SpaceError::InvalidConfiguration {
            description: format!(
                "Unknown VectorDB backend: '{}'. Available backends: {:?}",
                name,
                self.names()
            ),
        })?;
        factory.create(dimensions, config)
    }

    /// Get all registered backend names (unique, sorted).
    ///
    /// This returns only canonical names, not aliases.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .factories
            .values()
            .map(|f| f.name().to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Get all registered names including aliases.
    pub fn all_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for BackendRegistry {
    /// Create a registry with built-in backends pre-registered.
    ///
    /// This is equivalent to calling `BackendRegistry::with_defaults()`.
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock backend factory for testing
    struct MockBackendFactory {
        name: &'static str,
        aliases: Vec<&'static str>,
    }

    impl VectorBackendFactory for MockBackendFactory {
        fn name(&self) -> &str {
            self.name
        }

        fn aliases(&self) -> &[&str] {
            &self.aliases
        }

        fn create(
            &self,
            dimensions: usize,
            _config: &BackendConfig,
        ) -> Result<Box<dyn VectorBackendDyn>, SpaceError> {
            Ok(Box::new(MockBackend { dimensions }))
        }
    }

    /// Mock backend for testing
    struct MockBackend {
        dimensions: usize,
    }

    impl VectorBackendDyn for MockBackend {
        fn store(&mut self, _embedding: &[f32]) -> Result<usize, SpaceError> {
            Ok(0)
        }
        fn get(&self, _id: usize) -> Option<Vec<f32>> {
            None
        }
        fn remove(&mut self, _id: usize) -> bool {
            false
        }
        fn dimensions(&self) -> usize {
            self.dimensions
        }
        fn len(&self) -> usize {
            0
        }
        fn clear(&mut self) {}
        fn query(
            &self,
            _embedding: &[f32],
            _similarity_fn: Option<&str>,
            _threshold: Option<f32>,
            _ranking_fn: Option<&str>,
            _params: &[ResolvedArg],
        ) -> Result<Vec<(usize, f32)>, SpaceError> {
            Ok(vec![])
        }
        fn supported_similarity_fns(&self) -> Vec<String> {
            vec!["cosine".to_string()]
        }
        fn supported_ranking_fns(&self) -> Vec<String> {
            vec!["all".to_string()]
        }
        fn clone_boxed(&self) -> Box<dyn VectorBackendDyn> {
            Box::new(MockBackend { dimensions: self.dimensions })
        }
        fn all_ids(&self) -> Vec<usize> {
            vec![]
        }
    }

    #[test]
    fn test_registry_new_is_empty() {
        let registry = BackendRegistry::new();
        assert!(registry.names().is_empty());
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = BackendRegistry::new();

        let factory = Arc::new(MockBackendFactory {
            name: "test",
            aliases: vec!["t", "testing"],
        });

        registry.register(factory);

        // Primary name
        assert!(registry.contains("test"));
        assert!(registry.get("test").is_some());

        // Case insensitive
        assert!(registry.contains("TEST"));
        assert!(registry.contains("Test"));

        // Aliases
        assert!(registry.contains("t"));
        assert!(registry.contains("testing"));

        // Non-existent
        assert!(!registry.contains("unknown"));
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_registry_create_backend() {
        let mut registry = BackendRegistry::new();

        registry.register(Arc::new(MockBackendFactory {
            name: "mock",
            aliases: vec![],
        }));

        let backend = registry
            .create("mock", 128, &BackendConfig::default())
            .expect("create should succeed");

        assert_eq!(backend.dimensions(), 128);
    }

    #[test]
    fn test_registry_create_unknown_backend() {
        let registry = BackendRegistry::new();

        let result = registry.create("unknown", 128, &BackendConfig::default());

        assert!(result.is_err());
        if let Err(SpaceError::InvalidConfiguration { description }) = result {
            assert!(description.contains("Unknown VectorDB backend"));
            assert!(description.contains("unknown"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[test]
    fn test_registry_names() {
        let mut registry = BackendRegistry::new();

        registry.register(Arc::new(MockBackendFactory {
            name: "alpha",
            aliases: vec!["a"],
        }));

        registry.register(Arc::new(MockBackendFactory {
            name: "beta",
            aliases: vec!["b"],
        }));

        let names = registry.names();
        assert_eq!(names, vec!["alpha", "beta"]);

        let all_names = registry.all_names();
        assert!(all_names.contains(&"alpha".to_string()));
        assert!(all_names.contains(&"beta".to_string()));
        assert!(all_names.contains(&"a".to_string()));
        assert!(all_names.contains(&"b".to_string()));
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert!(config.api_key.is_none());
        assert!(config.endpoint.is_none());
        assert!(config.options.is_empty());
    }

    #[test]
    fn test_backend_config_with_values() {
        let mut options = HashMap::new();
        options.insert("timeout".to_string(), "30".to_string());

        let config = BackendConfig {
            api_key: Some("pk-xxx".to_string()),
            endpoint: Some("https://example.com".to_string()),
            options,
        };

        assert_eq!(config.api_key.as_deref(), Some("pk-xxx"));
        assert_eq!(config.endpoint.as_deref(), Some("https://example.com"));
        assert_eq!(config.options.get("timeout"), Some(&"30".to_string()));
    }

    #[test]
    fn test_resolved_arg_accessors() {
        let int_arg = ResolvedArg::Int(42);
        assert_eq!(int_arg.as_int(), Some(42));
        assert_eq!(int_arg.as_float(), None);

        let float_arg = ResolvedArg::Float(3.14);
        assert_eq!(float_arg.as_float(), Some(3.14));
        assert_eq!(float_arg.as_int(), None);

        let str_arg = ResolvedArg::String("hello".to_string());
        assert_eq!(str_arg.as_str(), Some("hello"));
        assert_eq!(str_arg.as_bool(), None);

        let bool_arg = ResolvedArg::Bool(true);
        assert_eq!(bool_arg.as_bool(), Some(true));
        assert_eq!(bool_arg.as_str(), None);
    }

    #[test]
    fn test_registry_with_defaults_has_rho_backend() {
        let registry = BackendRegistry::with_defaults();

        // Canonical name should be registered
        assert!(registry.contains("rho"));

        // Should be able to get the factory
        let factory = registry.get("rho");
        assert!(factory.is_some());
        assert_eq!(factory.unwrap().name(), "rho");

        // Should be able to create a backend
        let backend = registry.create("rho", 128, &BackendConfig::default());
        assert!(backend.is_ok());
        assert_eq!(backend.unwrap().dimensions(), 128);
    }

    #[test]
    fn test_registry_with_defaults_has_aliases() {
        let registry = BackendRegistry::with_defaults();

        // Common aliases should all work
        assert!(registry.contains("default"));
        assert!(registry.contains("memory"));
        assert!(registry.contains("inmemory"));

        // Case-insensitive aliases
        assert!(registry.contains("DEFAULT"));
        assert!(registry.contains("InMemory"));

        // All aliases should create the same backend type
        let backend_rho = registry.create("rho", 64, &BackendConfig::default()).unwrap();
        let backend_default = registry.create("default", 64, &BackendConfig::default()).unwrap();
        let backend_memory = registry.create("memory", 64, &BackendConfig::default()).unwrap();

        // All should have the same dimensions
        assert_eq!(backend_rho.dimensions(), 64);
        assert_eq!(backend_default.dimensions(), 64);
        assert_eq!(backend_memory.dimensions(), 64);

        // All should support the same similarity functions
        assert_eq!(
            backend_rho.supported_similarity_fns(),
            backend_default.supported_similarity_fns()
        );
    }

    #[test]
    fn test_registry_default_equals_with_defaults() {
        let registry_default = BackendRegistry::default();
        let registry_with_defaults = BackendRegistry::with_defaults();

        // Both should have the same registered backends
        assert_eq!(registry_default.names(), registry_with_defaults.names());
        assert_eq!(registry_default.all_names(), registry_with_defaults.all_names());

        // Both should be able to create the same backend
        assert!(registry_default.contains("rho"));
        assert!(registry_with_defaults.contains("rho"));
    }

    #[test]
    fn test_registry_new_vs_default() {
        let registry_new = BackendRegistry::new();
        let registry_default = BackendRegistry::default();

        // new() creates empty registry
        assert!(registry_new.names().is_empty());
        assert!(!registry_new.contains("rho"));

        // default() creates pre-populated registry
        assert!(!registry_default.names().is_empty());
        assert!(registry_default.contains("rho"));
    }
}
