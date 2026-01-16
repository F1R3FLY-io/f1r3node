//! Rholang integration module for rho-vectordb.
//!
//! This module provides the integration layer that allows rho-vectordb to register
//! its backends with rholang's BackendRegistry.
//!
//! # Usage
//!
//! ```ignore
//! use rholang::spaces::BackendRegistry;
//!
//! let mut registry = BackendRegistry::new();
//! rho_vectordb::register_with_rholang(&mut registry);
//!
//! // Now "rho", "default", "memory", "inmemory" backends are available
//! let backend = registry.create("rho", 384, &BackendConfig::default())?;
//! ```

use std::sync::Arc;

use crate::rust::interpreter::spaces::{
    BackendConfig, BackendRegistry, SpaceError, VectorBackendDyn, VectorBackendFactory, ResolvedArg,
};

use super::backend::VectorBackend;
use super::backend::InMemoryBackend;
use super::metrics::{IndexConfig, SimilarityMetric};
use super::error::VectorDBError;
use super::super::types::EmbeddingType;

// =============================================================================
// Helper: Convert VectorDBError to SpaceError
// =============================================================================

fn convert_vectordb_error(e: VectorDBError) -> SpaceError {
    match e {
        VectorDBError::DimensionMismatch { expected, actual } => {
            SpaceError::InvalidConfiguration {
                description: format!(
                    "Embedding dimension mismatch: expected {} dimensions, got {}",
                    expected, actual
                ),
            }
        }
        VectorDBError::InvalidConfiguration { description } => {
            SpaceError::InvalidConfiguration { description }
        }
        _ => SpaceError::InternalError {
            description: e.to_string(),
        },
    }
}

// =============================================================================
// VectorBackendFactory Implementation
// =============================================================================

/// Factory for creating rho-vectordb in-memory backends.
///
/// This factory creates `InMemoryBackend` instances wrapped in a
/// `VectorBackendDyn` implementation for use with rholang's VectorDB spaces.
pub struct RhoBackendFactory;

impl VectorBackendFactory for RhoBackendFactory {
    fn name(&self) -> &str {
        "rho"
    }

    fn aliases(&self) -> &[&str] {
        &["default", "memory", "inmemory"]
    }

    fn create(
        &self,
        dimensions: usize,
        config: &BackendConfig,
    ) -> Result<Box<dyn VectorBackendDyn>, SpaceError> {
        // Parse embedding_type from config (default: Float)
        let embedding_type = config
            .options
            .get("embedding_type")
            .and_then(|s| EmbeddingType::from_str(s))
            .unwrap_or(EmbeddingType::Float);

        // Parse metric from config if provided, otherwise derive from embedding_type
        // This makes the backend the single source of truth for default metrics.
        let metric = config
            .options
            .get("metric")
            .and_then(|s| SimilarityMetric::from_str(s))
            .unwrap_or_else(|| match embedding_type {
                EmbeddingType::Boolean => SimilarityMetric::Hamming,
                EmbeddingType::Integer | EmbeddingType::Float => SimilarityMetric::Cosine,
            });

        // Parse threshold from config if provided
        let threshold = config
            .options
            .get("threshold")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.8);

        // Configure index optimizations based on the metric.
        // Different metrics benefit from different optimizations:
        // - Cosine/DotProduct: pre-normalization (L2-normalized embeddings make cosine = dot product)
        // - Hamming/Jaccard: binary packing (for hardware popcnt acceleration)
        // - Euclidean: norm caching (for fast ||a-b||² computation)
        // - Manhattan: no special optimization needed
        let index_config = match metric {
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct => {
                // Pre-normalize embeddings for fast cosine similarity
                IndexConfig {
                    pre_normalize: true,
                    cache_norms: false,
                    pack_binary: false,
                    hnsw: None,
                    scalar_quantization: None,
                }
            }
            SimilarityMetric::Hamming | SimilarityMetric::Jaccard => {
                // Pack binary for hardware-accelerated Hamming/Jaccard
                // Don't pre-normalize (destroys binary structure)
                IndexConfig {
                    pre_normalize: false,
                    cache_norms: false,
                    pack_binary: true,
                    hnsw: None,
                    scalar_quantization: None,
                }
            }
            SimilarityMetric::Euclidean => {
                // Cache squared norms for fast Euclidean distance
                // Don't pre-normalize (changes distance values)
                IndexConfig {
                    pre_normalize: false,
                    cache_norms: true,
                    pack_binary: false,
                    hnsw: None,
                    scalar_quantization: None,
                }
            }
            SimilarityMetric::Manhattan => {
                // No special optimization for Manhattan
                IndexConfig {
                    pre_normalize: false,
                    cache_norms: false,
                    pack_binary: false,
                    hnsw: None,
                    scalar_quantization: None,
                }
            }
        };

        let backend = InMemoryBackend::with_index_config(dimensions, index_config);

        Ok(Box::new(InMemoryBackendWrapper {
            inner: backend,
            metric,
            threshold,
        }))
    }
}

// =============================================================================
// VectorBackendDyn Implementation
// =============================================================================

/// Wrapper around `InMemoryBackend` implementing `VectorBackendDyn`.
///
/// This wrapper adapts the concrete `InMemoryBackend` to the object-safe
/// `VectorBackendDyn` trait that rholang uses for dynamic dispatch.
struct InMemoryBackendWrapper {
    inner: InMemoryBackend,
    metric: SimilarityMetric,
    threshold: f32,
}

impl VectorBackendDyn for InMemoryBackendWrapper {
    fn store(&mut self, embedding: &[f32]) -> Result<usize, SpaceError> {
        self.inner
            .store(embedding)
            .map_err(convert_vectordb_error)
    }

    fn get(&self, id: usize) -> Option<Vec<f32>> {
        self.inner.get(&id)
    }

    fn remove(&mut self, id: usize) -> bool {
        self.inner.remove(&id)
    }

    fn dimensions(&self) -> usize {
        VectorBackend::dimensions(&self.inner)
    }

    fn len(&self) -> usize {
        VectorBackend::len(&self.inner)
    }

    fn clear(&mut self) {
        VectorBackend::clear(&mut self.inner);
    }

    fn query(
        &self,
        embedding: &[f32],
        similarity_fn: Option<&str>,
        threshold: Option<f32>,
        ranking_fn: Option<&str>,
        params: &[ResolvedArg],
    ) -> Result<Vec<(usize, f32)>, SpaceError> {
        // Parse similarity function (or use default)
        let metric = similarity_fn
            .and_then(SimilarityMetric::from_str)
            .unwrap_or(self.metric);

        // Use threshold from params or default
        let thresh = threshold.unwrap_or(self.threshold);

        // Get top_k from params if ranking_fn is "top_k"
        let top_k = if ranking_fn == Some("top_k") || ranking_fn == Some("topk") {
            params
                .first()
                .and_then(|p| match p {
                    ResolvedArg::Int(k) => Some(*k as usize),
                    ResolvedArg::Float(k) => Some(*k as usize),
                    _ => None,
                })
        } else {
            None
        };

        // Query the backend using find_similar
        self.inner
            .find_similar(embedding, metric, thresh, top_k)
            .map_err(convert_vectordb_error)
    }

    fn default_threshold(&self) -> f32 {
        self.threshold
    }

    fn default_similarity_fn(&self) -> &str {
        self.metric.as_str()
    }

    fn default_ranking_fn(&self) -> &str {
        "all"
    }

    fn supported_similarity_fns(&self) -> Vec<String> {
        vec![
            "cosine".to_string(),
            "dot_product".to_string(),
            "euclidean".to_string(),
            "manhattan".to_string(),
            "hamming".to_string(),
            "jaccard".to_string(),
        ]
    }

    fn supported_ranking_fns(&self) -> Vec<String> {
        vec![
            "all".to_string(),
            "top_k".to_string(),
            "topk".to_string(),
        ]
    }

    fn clone_boxed(&self) -> Box<dyn VectorBackendDyn> {
        Box::new(InMemoryBackendWrapper {
            inner: self.inner.clone(),
            metric: self.metric,
            threshold: self.threshold,
        })
    }

    fn all_ids(&self) -> Vec<usize> {
        // Get all valid IDs from the backend's id_to_row map
        self.inner.all_ids()
    }
}

// =============================================================================
// Registration Function
// =============================================================================

/// Register rho-vectordb backends with a rholang BackendRegistry.
///
/// This function registers the following backends:
/// - `"rho"` (canonical name)
/// - `"default"` (alias)
/// - `"memory"` (alias)
/// - `"inmemory"` (alias)
///
/// # Example
///
/// ```ignore
/// use rholang::spaces::BackendRegistry;
///
/// let mut registry = BackendRegistry::new();
/// rho_vectordb::register_with_rholang(&mut registry);
///
/// // Create a backend
/// let backend = registry.create("rho", 384, &BackendConfig::default())?;
/// ```
pub fn register_with_rholang(registry: &mut BackendRegistry) {
    registry.register(Arc::new(RhoBackendFactory));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rho_backend_factory_name() {
        let factory = RhoBackendFactory;
        assert_eq!(factory.name(), "rho");
        assert_eq!(factory.aliases(), &["default", "memory", "inmemory"]);
    }

    #[test]
    fn test_create_backend() {
        let factory = RhoBackendFactory;
        let backend = factory
            .create(384, &BackendConfig::default())
            .expect("Failed to create backend");

        assert_eq!(backend.dimensions(), 384);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[test]
    fn test_store_and_query() {
        let factory = RhoBackendFactory;
        let mut backend = factory
            .create(3, &BackendConfig::default())
            .expect("Failed to create backend");

        // Store some embeddings
        let id1 = backend.store(&[1.0, 0.0, 0.0]).expect("Failed to store");
        let id2 = backend.store(&[0.9, 0.1, 0.0]).expect("Failed to store");
        let _id3 = backend.store(&[0.0, 1.0, 0.0]).expect("Failed to store");

        assert_eq!(backend.len(), 3);

        // Query for similar embeddings
        let results = backend
            .query(&[1.0, 0.0, 0.0], Some("cosine"), Some(0.5), None, &[])
            .expect("Query failed");

        // Should find id1 (exact match) and id2 (similar)
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id1); // Best match should be id1
    }
}
