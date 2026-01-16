//! VectorDB configuration parsing from generic RawConfigValue types.
//!
//! This module is responsible for interpreting VectorDB-specific configuration
//! parameters from the generic `RawConfigValue` types passed through from Rholang.
//!
//! # Architecture
//!
//! The Rholang interpreter extracts only `dimensions` (the universal parameter)
//! and passes everything else as generic key-value pairs. This module parses
//! those key-value pairs into VectorDB-specific types:
//!
//! - `threshold`: Similarity threshold (0.0-1.0)
//! - `metric`: Similarity metric (cosine, euclidean, etc.)
//! - `embedding_type`: Embedding format (boolean, integer, float)
//! - `index` / `indices`: Index optimization configuration
//!
//! # Example
//!
//! ```ignore
//! use rho_vectordb::db::config::{VectorDBConfig, parse_config};
//!
//! let raw = RawVectorDBConfig::new(384);
//! raw.insert("threshold", RawConfigValue::String("0.7".to_string()));
//! raw.insert("metric", RawConfigValue::String("cosine".to_string()));
//! raw.insert("index", RawConfigValue::String("pre_normalize".to_string()));
//!
//! let config = parse_config(raw)?;
//! assert_eq!(config.dimensions, 384);
//! assert_eq!(config.threshold, 0.7);
//! assert!(config.index_config.pre_normalize);
//! ```

use std::collections::HashMap;

use super::super::error::VectorDBError;
use super::super::metrics::{
    EmbeddingType, HnswConfig, IndexConfig, IndexType, ScalarQuantizationConfig, SimilarityMetric,
};

// =============================================================================
// RawConfigValue Re-definition (for VectorDB-side parsing)
// =============================================================================

/// A generic configuration value for VectorDB configuration.
///
/// This mirrors the `RawConfigValue` from Rholang to avoid cross-crate
/// dependencies. Values are converted from Rholang's representation.
#[derive(Debug, Clone)]
pub enum RawConfigValue {
    /// A string value
    String(String),
    /// An integer value
    Int(i64),
    /// A floating-point value
    Float(f64),
    /// A boolean value
    Bool(bool),
    /// A list of values
    List(Vec<RawConfigValue>),
    /// A nested map
    Map(HashMap<String, RawConfigValue>),
}

impl RawConfigValue {
    /// Try to get as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RawConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            RawConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to parse as a float (handles string conversion).
    pub fn try_as_float(&self) -> Option<f64> {
        match self {
            RawConfigValue::Float(f) => Some(*f),
            RawConfigValue::Int(i) => Some(*i as f64),
            RawConfigValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to get as a list reference.
    pub fn as_list(&self) -> Option<&[RawConfigValue]> {
        match self {
            RawConfigValue::List(list) => Some(list),
            _ => None,
        }
    }

    /// Try to get as a map reference.
    pub fn as_map(&self) -> Option<&HashMap<String, RawConfigValue>> {
        match self {
            RawConfigValue::Map(map) => Some(map),
            _ => None,
        }
    }
}

// =============================================================================
// VectorDBConfig
// =============================================================================

/// Fully parsed VectorDB configuration.
///
/// This struct contains all VectorDB parameters after parsing from the
/// generic `RawConfigValue` representation.
#[derive(Debug, Clone)]
pub struct VectorDBConfig {
    /// Required: embedding dimensions.
    pub dimensions: usize,

    /// Similarity threshold (0.0-1.0). Default: 0.8
    pub threshold: f32,

    /// Similarity metric. Default: derived from embedding_type
    pub metric: SimilarityMetric,

    /// Embedding type/format. Default: Float
    pub embedding_type: EmbeddingType,

    /// Index optimization configuration.
    pub index_config: IndexConfig,

    /// Optional initial capacity hint.
    pub capacity: Option<usize>,
}

impl Default for VectorDBConfig {
    fn default() -> Self {
        Self {
            dimensions: 0, // Must be set
            threshold: 0.8,
            metric: SimilarityMetric::Cosine,
            embedding_type: EmbeddingType::Float,
            index_config: IndexConfig::default(),
            capacity: None,
        }
    }
}

impl VectorDBConfig {
    /// Create a new config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Set the similarity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the similarity metric.
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the embedding type.
    pub fn with_embedding_type(mut self, embedding_type: EmbeddingType) -> Self {
        self.embedding_type = embedding_type;
        self
    }

    /// Set the index configuration.
    pub fn with_index_config(mut self, index_config: IndexConfig) -> Self {
        self.index_config = index_config;
        self
    }

    /// Set the capacity hint.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }
}

// =============================================================================
// Configuration Parsing
// =============================================================================

/// Parse a VectorDBConfig from dimensions and a params map.
///
/// This is the main entry point for parsing VectorDB configuration from
/// the generic representation passed through from Rholang.
///
/// # Arguments
///
/// * `dimensions` - The required embedding dimensions
/// * `params` - Optional configuration parameters as key-value pairs
///
/// # Returns
///
/// A fully parsed `VectorDBConfig` or an error if parsing fails.
pub fn parse_config(
    dimensions: usize,
    params: &HashMap<String, RawConfigValue>,
) -> Result<VectorDBConfig, VectorDBError> {
    let mut config = VectorDBConfig::new(dimensions);

    // Parse threshold
    if let Some(value) = params.get("threshold") {
        config.threshold = parse_threshold(value)?;
    }

    // Parse embedding_type first (metric defaults depend on it)
    if let Some(value) = params.get("embedding_type") {
        config.embedding_type = parse_embedding_type(value)?;
    }

    // Parse metric (defaults based on embedding_type)
    if let Some(value) = params.get("metric") {
        config.metric = parse_metric(value)?;
    } else {
        // Derive default metric from embedding type
        config.metric = match config.embedding_type {
            EmbeddingType::Boolean => SimilarityMetric::Hamming,
            EmbeddingType::Integer | EmbeddingType::Float => SimilarityMetric::Cosine,
        };
    }

    // Parse capacity
    if let Some(value) = params.get("capacity") {
        if let Some(cap) = value.as_int() {
            if cap > 0 {
                config.capacity = Some(cap as usize);
            }
        }
    }

    // Parse index configuration
    if let Some(value) = params.get("index") {
        config.index_config = parse_index_config(value)?;
    } else if let Some(value) = params.get("indices") {
        config.index_config = parse_index_config(value)?;
    }

    Ok(config)
}

/// Parse threshold from a RawConfigValue.
///
/// Accepts:
/// - Float: 0.7 -> 0.7
/// - String: "0.7" -> 0.7
/// - Integer (0-100): 70 -> 0.7
fn parse_threshold(value: &RawConfigValue) -> Result<f32, VectorDBError> {
    match value {
        RawConfigValue::Float(f) => {
            if !(0.0..=1.0).contains(f) {
                return Err(VectorDBError::InvalidConfiguration {
                    description: "Threshold must be between 0.0 and 1.0".to_string(),
                });
            }
            Ok(*f as f32)
        }
        RawConfigValue::String(s) => {
            let f: f64 = s.parse().map_err(|_| VectorDBError::InvalidConfiguration {
                description: format!("Cannot parse threshold '{}' as float", s),
            })?;
            if !(0.0..=1.0).contains(&f) {
                return Err(VectorDBError::InvalidConfiguration {
                    description: "Threshold must be between 0.0 and 1.0".to_string(),
                });
            }
            Ok(f as f32)
        }
        RawConfigValue::Int(i) => {
            if !(0..=100).contains(i) {
                return Err(VectorDBError::InvalidConfiguration {
                    description: "Threshold (integer) must be between 0 and 100".to_string(),
                });
            }
            Ok((*i as f32) / 100.0)
        }
        _ => Err(VectorDBError::InvalidConfiguration {
            description: "Threshold must be a float, string, or integer".to_string(),
        }),
    }
}

/// Parse embedding_type from a RawConfigValue.
fn parse_embedding_type(value: &RawConfigValue) -> Result<EmbeddingType, VectorDBError> {
    match value {
        RawConfigValue::String(s) => {
            EmbeddingType::from_str(s).ok_or_else(|| VectorDBError::InvalidConfiguration {
                description: format!(
                    "Unknown embedding_type '{}'. Use: boolean, integer, or float",
                    s
                ),
            })
        }
        _ => Err(VectorDBError::InvalidConfiguration {
            description: "embedding_type must be a string".to_string(),
        }),
    }
}

/// Parse metric from a RawConfigValue.
fn parse_metric(value: &RawConfigValue) -> Result<SimilarityMetric, VectorDBError> {
    match value {
        RawConfigValue::String(s) => {
            SimilarityMetric::from_str(s).ok_or_else(|| VectorDBError::InvalidConfiguration {
                description: format!(
                    "Unknown metric '{}'. Use: cosine, dot, euclidean, manhattan, hamming, or jaccard",
                    s
                ),
            })
        }
        _ => Err(VectorDBError::InvalidConfiguration {
            description: "metric must be a string".to_string(),
        }),
    }
}

/// Parse index configuration from a RawConfigValue.
///
/// Accepts:
/// - String: "pre_normalize" -> IndexConfig with pre_normalize = true
/// - List: ["pre_normalize", "cache_norms"] -> IndexConfig with both enabled
/// - Map: {"type": "hnsw", "max_connections": 32} -> IndexConfig with HNSW
fn parse_index_config(value: &RawConfigValue) -> Result<IndexConfig, VectorDBError> {
    let mut config = IndexConfig::default();

    match value {
        RawConfigValue::String(s) => {
            apply_index_name(&mut config, s)?;
        }
        RawConfigValue::List(items) => {
            for item in items {
                match item {
                    RawConfigValue::String(s) => {
                        apply_index_name(&mut config, s)?;
                    }
                    RawConfigValue::Map(m) => {
                        apply_index_map(&mut config, m)?;
                    }
                    _ => {
                        return Err(VectorDBError::InvalidConfiguration {
                            description: "Index list items must be strings or maps".to_string(),
                        });
                    }
                }
            }
        }
        RawConfigValue::Map(m) => {
            apply_index_map(&mut config, m)?;
        }
        _ => {
            return Err(VectorDBError::InvalidConfiguration {
                description: "Index must be a string, list, or map".to_string(),
            });
        }
    }

    Ok(config)
}

/// Apply an index by name string.
fn apply_index_name(config: &mut IndexConfig, name: &str) -> Result<(), VectorDBError> {
    match IndexType::from_str(name) {
        Some(IndexType::PreNormalize) => {
            config.pre_normalize = true;
        }
        Some(IndexType::CacheNorms) => {
            config.cache_norms = true;
        }
        Some(IndexType::PackBinary) => {
            config.pack_binary = true;
        }
        Some(IndexType::Hnsw) => {
            config.hnsw = Some(HnswConfig::default());
        }
        Some(IndexType::ScalarQuantization) => {
            config.scalar_quantization = Some(ScalarQuantizationConfig::default());
        }
        None => {
            return Err(VectorDBError::InvalidConfiguration {
                description: format!(
                    "Unknown index type '{}'. Use: pre_normalize, cache_norms, pack_binary, hnsw, or sq8",
                    name
                ),
            });
        }
    }
    Ok(())
}

/// Apply index configuration from a map.
fn apply_index_map(
    config: &mut IndexConfig,
    map: &HashMap<String, RawConfigValue>,
) -> Result<(), VectorDBError> {
    // Get the type field
    let type_name = map
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| VectorDBError::InvalidConfiguration {
            description: "Index map must have a 'type' field".to_string(),
        })?;

    match IndexType::from_str(type_name) {
        Some(IndexType::PreNormalize) => {
            config.pre_normalize = true;
        }
        Some(IndexType::CacheNorms) => {
            config.cache_norms = true;
        }
        Some(IndexType::PackBinary) => {
            config.pack_binary = true;
        }
        Some(IndexType::Hnsw) => {
            config.hnsw = Some(parse_hnsw_config(map)?);
        }
        Some(IndexType::ScalarQuantization) => {
            config.scalar_quantization = Some(parse_scalar_quantization_config(map)?);
        }
        None => {
            return Err(VectorDBError::InvalidConfiguration {
                description: format!(
                    "Unknown index type '{}'. Use: pre_normalize, cache_norms, pack_binary, hnsw, or sq8",
                    type_name
                ),
            });
        }
    }

    Ok(())
}

/// Parse HNSW configuration from a map.
fn parse_hnsw_config(map: &HashMap<String, RawConfigValue>) -> Result<HnswConfig, VectorDBError> {
    let mut config = HnswConfig::default();

    if let Some(value) = map.get("max_connections") {
        if let Some(n) = value.as_int() {
            if n > 0 {
                config.max_connections = n as usize;
            }
        }
    }

    if let Some(value) = map.get("ef_construction") {
        if let Some(n) = value.as_int() {
            if n > 0 {
                config.ef_construction = n as usize;
            }
        }
    }

    if let Some(value) = map.get("ef_search") {
        if let Some(n) = value.as_int() {
            if n > 0 {
                config.ef_search = n as usize;
            }
        }
    }

    Ok(config)
}

/// Parse scalar quantization configuration from a map.
fn parse_scalar_quantization_config(
    map: &HashMap<String, RawConfigValue>,
) -> Result<ScalarQuantizationConfig, VectorDBError> {
    let mut config = ScalarQuantizationConfig::default();

    if let Some(value) = map.get("bits") {
        if let Some(n) = value.as_int() {
            if n == 1 || n == 2 || n == 4 || n == 8 {
                config.bits = n as u8;
            } else {
                return Err(VectorDBError::InvalidConfiguration {
                    description: "Scalar quantization bits must be 1, 2, 4, or 8".to_string(),
                });
            }
        }
    }

    if let Some(value) = map.get("keep_original") {
        if let RawConfigValue::Bool(b) = value {
            config.keep_original = *b;
        }
    }

    Ok(config)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> HashMap<String, RawConfigValue> {
        HashMap::new()
    }

    #[test]
    fn test_parse_config_minimal() {
        let config = parse_config(384, &make_params()).expect("should parse");
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.threshold, 0.8); // default
        assert_eq!(config.metric, SimilarityMetric::Cosine); // default for Float
        assert_eq!(config.embedding_type, EmbeddingType::Float); // default
        assert!(!config.index_config.has_any_optimization());
    }

    #[test]
    fn test_parse_config_with_threshold_float() {
        let mut params = make_params();
        params.insert("threshold".to_string(), RawConfigValue::Float(0.7));

        let config = parse_config(128, &params).expect("should parse");
        assert!((config.threshold - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_parse_config_with_threshold_string() {
        let mut params = make_params();
        params.insert(
            "threshold".to_string(),
            RawConfigValue::String("0.65".to_string()),
        );

        let config = parse_config(128, &params).expect("should parse");
        assert!((config.threshold - 0.65).abs() < 0.001);
    }

    #[test]
    fn test_parse_config_with_threshold_int() {
        let mut params = make_params();
        params.insert("threshold".to_string(), RawConfigValue::Int(75));

        let config = parse_config(128, &params).expect("should parse");
        assert!((config.threshold - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_parse_config_with_metric() {
        let mut params = make_params();
        params.insert(
            "metric".to_string(),
            RawConfigValue::String("euclidean".to_string()),
        );

        let config = parse_config(128, &params).expect("should parse");
        assert_eq!(config.metric, SimilarityMetric::Euclidean);
    }

    #[test]
    fn test_parse_config_with_embedding_type() {
        let mut params = make_params();
        params.insert(
            "embedding_type".to_string(),
            RawConfigValue::String("boolean".to_string()),
        );

        let config = parse_config(64, &params).expect("should parse");
        assert_eq!(config.embedding_type, EmbeddingType::Boolean);
        // Metric should default to Hamming for Boolean
        assert_eq!(config.metric, SimilarityMetric::Hamming);
    }

    #[test]
    fn test_parse_config_with_single_index() {
        let mut params = make_params();
        params.insert(
            "index".to_string(),
            RawConfigValue::String("pre_normalize".to_string()),
        );

        let config = parse_config(384, &params).expect("should parse");
        assert!(config.index_config.pre_normalize);
        assert!(!config.index_config.cache_norms);
    }

    #[test]
    fn test_parse_config_with_multiple_indices() {
        let mut params = make_params();
        params.insert(
            "indices".to_string(),
            RawConfigValue::List(vec![
                RawConfigValue::String("pre_normalize".to_string()),
                RawConfigValue::String("cache_norms".to_string()),
            ]),
        );

        let config = parse_config(384, &params).expect("should parse");
        assert!(config.index_config.pre_normalize);
        assert!(config.index_config.cache_norms);
    }

    #[test]
    fn test_parse_config_with_hnsw_map() {
        let mut hnsw_map = HashMap::new();
        hnsw_map.insert("type".to_string(), RawConfigValue::String("hnsw".to_string()));
        hnsw_map.insert("max_connections".to_string(), RawConfigValue::Int(32));
        hnsw_map.insert("ef_construction".to_string(), RawConfigValue::Int(400));

        let mut params = make_params();
        params.insert("index".to_string(), RawConfigValue::Map(hnsw_map));

        let config = parse_config(768, &params).expect("should parse");
        assert!(config.index_config.hnsw.is_some());
        let hnsw = config.index_config.hnsw.expect("HNSW should exist");
        assert_eq!(hnsw.max_connections, 32);
        assert_eq!(hnsw.ef_construction, 400);
    }

    #[test]
    fn test_parse_config_with_mixed_indices() {
        let mut hnsw_map = HashMap::new();
        hnsw_map.insert("type".to_string(), RawConfigValue::String("hnsw".to_string()));
        hnsw_map.insert("max_connections".to_string(), RawConfigValue::Int(16));

        let mut params = make_params();
        params.insert(
            "indices".to_string(),
            RawConfigValue::List(vec![
                RawConfigValue::String("pre_normalize".to_string()),
                RawConfigValue::Map(hnsw_map),
            ]),
        );

        let config = parse_config(384, &params).expect("should parse");
        assert!(config.index_config.pre_normalize);
        assert!(config.index_config.hnsw.is_some());
    }

    #[test]
    fn test_parse_threshold_out_of_range() {
        let mut params = make_params();
        params.insert("threshold".to_string(), RawConfigValue::Float(1.5));

        let result = parse_config(128, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unknown_metric() {
        let mut params = make_params();
        params.insert(
            "metric".to_string(),
            RawConfigValue::String("unknown_metric".to_string()),
        );

        let result = parse_config(128, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unknown_index() {
        let mut params = make_params();
        params.insert(
            "index".to_string(),
            RawConfigValue::String("unknown_index".to_string()),
        );

        let result = parse_config(128, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_vectordb_config_builder() {
        let config = VectorDBConfig::new(256)
            .with_threshold(0.9)
            .with_metric(SimilarityMetric::DotProduct)
            .with_embedding_type(EmbeddingType::Integer)
            .with_capacity(10000);

        assert_eq!(config.dimensions, 256);
        assert!((config.threshold - 0.9).abs() < 0.001);
        assert_eq!(config.metric, SimilarityMetric::DotProduct);
        assert_eq!(config.embedding_type, EmbeddingType::Integer);
        assert_eq!(config.capacity, Some(10000));
    }
}
