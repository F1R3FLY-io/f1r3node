//! Raw Configuration Types for VectorDB Pass-Through
//!
//! This module provides generic configuration types that allow Rholang to pass
//! configuration to VectorDB backends without understanding their semantics.
//!
//! # Design Principle
//!
//! Rholang is agnostic to VectorDB-specific semantics. The interpreter:
//! 1. Parses the Rholang configuration map into a generic `RawVectorDBConfig`
//! 2. Passes the entire configuration to the VectorDB backend
//! 3. The backend validates and interprets its own parameters
//!
//! This decoupling allows different VectorDB backends to accept different
//! configurations without coupling Rholang to any specific implementation.
//!
//! # Example
//!
//! ```rholang
//! VectorDBFactory!({
//!   "dimensions": 384,
//!   "metric": "cosine",
//!   "threshold": "0.7",
//!   "index": "pre_normalize"
//! }, *space)
//! ```
//!
//! Rholang extracts `dimensions` (the only universal parameter) and passes
//! everything else (`metric`, `threshold`, `index`) as generic key-value pairs
//! for the VectorDB backend to interpret.

use std::collections::HashMap;
use std::fmt;

use models::rhoapi::{expr::ExprInstance, Par};

// ==========================================================================
// RawConfigValue - Generic Configuration Value
// ==========================================================================

/// A generic configuration value that can represent any Rholang primitive or collection.
///
/// This type allows Rholang to pass configuration values to VectorDB backends
/// without interpreting their meaning. The backend is responsible for validating
/// and converting these values to their appropriate types.
///
/// # Supported Types
///
/// | Rholang Type | RawConfigValue Variant |
/// |--------------|------------------------|
/// | String       | `String(String)`       |
/// | Integer      | `Int(i64)`             |
/// | Float*       | `Float(f64)`           |
/// | Boolean      | `Bool(bool)`           |
/// | List         | `List(Vec<...>)`       |
/// | Map          | `Map(HashMap<...>)`    |
///
/// *Note: Rholang doesn't have native float literals; floats are typically
/// passed as strings like `"0.7"` and parsed by the backend.
#[derive(Debug, Clone, PartialEq)]
pub enum RawConfigValue {
    /// A string value (e.g., `"cosine"`, `"0.7"`)
    String(String),

    /// An integer value (e.g., `384`, `16`)
    Int(i64),

    /// A floating-point value (for backends that parse numeric strings)
    Float(f64),

    /// A boolean value (e.g., `true`, `false`)
    Bool(bool),

    /// A list of values (e.g., `["pre_normalize", "cache_norms"]`)
    List(Vec<RawConfigValue>),

    /// A nested map of key-value pairs (e.g., `{"type": "hnsw", "max_connections": 32}`)
    Map(HashMap<String, RawConfigValue>),
}

impl fmt::Display for RawConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawConfigValue::String(s) => write!(f, "\"{}\"", s),
            RawConfigValue::Int(i) => write!(f, "{}", i),
            RawConfigValue::Float(n) => write!(f, "{}", n),
            RawConfigValue::Bool(b) => write!(f, "{}", b),
            RawConfigValue::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            RawConfigValue::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl RawConfigValue {
    /// Try to get the value as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RawConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get the value as an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            RawConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get the value as a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            RawConfigValue::Float(f) => Some(*f),
            RawConfigValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get the value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            RawConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get the value as a list reference.
    pub fn as_list(&self) -> Option<&[RawConfigValue]> {
        match self {
            RawConfigValue::List(list) => Some(list),
            _ => None,
        }
    }

    /// Try to get the value as a map reference.
    pub fn as_map(&self) -> Option<&HashMap<String, RawConfigValue>> {
        match self {
            RawConfigValue::Map(map) => Some(map),
            _ => None,
        }
    }

    /// Check if this value is a string.
    pub fn is_string(&self) -> bool {
        matches!(self, RawConfigValue::String(_))
    }

    /// Check if this value is an integer.
    pub fn is_int(&self) -> bool {
        matches!(self, RawConfigValue::Int(_))
    }

    /// Check if this value is a float.
    pub fn is_float(&self) -> bool {
        matches!(self, RawConfigValue::Float(_))
    }

    /// Check if this value is a boolean.
    pub fn is_bool(&self) -> bool {
        matches!(self, RawConfigValue::Bool(_))
    }

    /// Check if this value is a list.
    pub fn is_list(&self) -> bool {
        matches!(self, RawConfigValue::List(_))
    }

    /// Check if this value is a map.
    pub fn is_map(&self) -> bool {
        matches!(self, RawConfigValue::Map(_))
    }

    /// Try to parse this value as a float, handling string conversion.
    ///
    /// This is useful for VectorDB backends that need to parse threshold values
    /// which may be passed as strings like `"0.7"`.
    pub fn try_as_float(&self) -> Option<f64> {
        match self {
            RawConfigValue::Float(f) => Some(*f),
            RawConfigValue::Int(i) => Some(*i as f64),
            RawConfigValue::String(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Try to parse this value as a usize, handling string conversion.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            RawConfigValue::Int(i) if *i >= 0 => Some(*i as usize),
            RawConfigValue::String(s) => s.parse::<usize>().ok(),
            _ => None,
        }
    }
}

// ==========================================================================
// RawVectorDBConfig - Top-Level Configuration
// ==========================================================================

/// Raw configuration extracted from Rholang for VectorDB construction.
///
/// This struct contains the minimal universal parameter (`dimensions`) that
/// all VectorDB implementations require, plus a generic `params` map for
/// backend-specific configuration.
///
/// # Rholang Extraction
///
/// Rholang only extracts `dimensions` from the configuration map. All other
/// parameters are passed through as-is in the `params` map:
///
/// ```rholang
/// VectorDBFactory!({
///   "dimensions": 384,      // Extracted by Rholang
///   "metric": "cosine",     // Passed to backend
///   "threshold": "0.7",     // Passed to backend
///   "index": "pre_normalize" // Passed to backend
/// }, *space)
/// ```
///
/// # Backend Interpretation
///
/// The VectorDB backend receives the `params` map and interprets:
/// - `metric`: Similarity metric (cosine, euclidean, etc.)
/// - `threshold`: Similarity threshold
/// - `embedding_type`: Data type of embeddings
/// - `index` / `indices`: Index optimization configuration
/// - Any other backend-specific parameters
#[derive(Debug, Clone, Default)]
pub struct RawVectorDBConfig {
    /// Required: embedding dimensions (the only universal parameter).
    ///
    /// This is the only parameter that Rholang understands. All VectorDB
    /// implementations need to know the dimensionality of embeddings.
    pub dimensions: usize,

    /// All other configuration as key-value pairs (passed to backend as-is).
    ///
    /// The VectorDB backend is responsible for validating and interpreting
    /// these parameters according to its own configuration schema.
    pub params: HashMap<String, RawConfigValue>,
}

impl RawVectorDBConfig {
    /// Create a new RawVectorDBConfig with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        RawVectorDBConfig {
            dimensions,
            params: HashMap::new(),
        }
    }

    /// Create a RawVectorDBConfig with dimensions and parameters.
    pub fn with_params(dimensions: usize, params: HashMap<String, RawConfigValue>) -> Self {
        RawVectorDBConfig { dimensions, params }
    }

    /// Get a parameter by key.
    pub fn get(&self, key: &str) -> Option<&RawConfigValue> {
        self.params.get(key)
    }

    /// Check if a parameter exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    /// Insert a parameter.
    pub fn insert(&mut self, key: impl Into<String>, value: RawConfigValue) {
        self.params.insert(key.into(), value);
    }

    /// Get a string parameter.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }

    /// Get an integer parameter.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.params.get(key).and_then(|v| v.as_int())
    }

    /// Get a float parameter, handling string conversion.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.try_as_float())
    }

    /// Get a boolean parameter.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(|v| v.as_bool())
    }
}

// ==========================================================================
// Par Conversion
// ==========================================================================

/// Error type for configuration parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawConfigError {
    /// A required parameter is missing.
    MissingRequired { parameter: String },

    /// A parameter has an invalid type.
    InvalidType { parameter: String, expected: String, actual: String },

    /// A parameter has an invalid value.
    InvalidValue { parameter: String, description: String },

    /// Failed to extract a string key from a map.
    InvalidMapKey { description: String },

    /// The configuration is not a map.
    NotAMap,

    /// General conversion error.
    ConversionError { description: String },
}

impl fmt::Display for RawConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawConfigError::MissingRequired { parameter } => {
                write!(f, "Missing required parameter: '{}'", parameter)
            }
            RawConfigError::InvalidType { parameter, expected, actual } => {
                write!(f, "Invalid type for '{}': expected {}, got {}", parameter, expected, actual)
            }
            RawConfigError::InvalidValue { parameter, description } => {
                write!(f, "Invalid value for '{}': {}", parameter, description)
            }
            RawConfigError::InvalidMapKey { description } => {
                write!(f, "Invalid map key: {}", description)
            }
            RawConfigError::NotAMap => {
                write!(f, "Configuration must be a map")
            }
            RawConfigError::ConversionError { description } => {
                write!(f, "Configuration conversion error: {}", description)
            }
        }
    }
}

impl std::error::Error for RawConfigError {}

/// Convert a Par to a RawConfigValue (recursive).
///
/// This function recursively converts a Rholang Par expression to a generic
/// RawConfigValue that can be passed to VectorDB backends.
pub fn par_to_raw_config_value(par: &Par) -> Result<RawConfigValue, RawConfigError> {
    // Check expressions (most common case for data)
    for expr in &par.exprs {
        if let Some(ref instance) = expr.expr_instance {
            return expr_instance_to_raw_config_value(instance);
        }
    }

    // Empty par or unsupported structure
    Err(RawConfigError::ConversionError {
        description: "Unsupported Par structure for configuration value".to_string(),
    })
}

/// Convert an ExprInstance to a RawConfigValue.
fn expr_instance_to_raw_config_value(instance: &ExprInstance) -> Result<RawConfigValue, RawConfigError> {
    match instance {
        ExprInstance::GString(s) => Ok(RawConfigValue::String(s.clone())),

        ExprInstance::GInt(i) => Ok(RawConfigValue::Int(*i)),

        ExprInstance::GBool(b) => Ok(RawConfigValue::Bool(*b)),

        ExprInstance::EListBody(elist) => {
            let items: Result<Vec<_>, _> = elist.ps
                .iter()
                .map(par_to_raw_config_value)
                .collect();
            Ok(RawConfigValue::List(items?))
        }

        ExprInstance::EMapBody(emap) => {
            let mut map = HashMap::new();
            for kv in &emap.kvs {
                // Extract key (must be a string)
                let key = if let Some(ref key_par) = kv.key {
                    extract_string_from_par(key_par)?
                } else {
                    return Err(RawConfigError::InvalidMapKey {
                        description: "Map key is None".to_string(),
                    });
                };

                // Extract value
                let value = if let Some(ref value_par) = kv.value {
                    par_to_raw_config_value(value_par)?
                } else {
                    return Err(RawConfigError::ConversionError {
                        description: format!("Map value for key '{}' is None", key),
                    });
                };

                map.insert(key, value);
            }
            Ok(RawConfigValue::Map(map))
        }

        // Tuples are converted to lists
        ExprInstance::ETupleBody(etuple) => {
            let items: Result<Vec<_>, _> = etuple.ps
                .iter()
                .map(par_to_raw_config_value)
                .collect();
            Ok(RawConfigValue::List(items?))
        }

        _ => Err(RawConfigError::ConversionError {
            description: format!("Unsupported expression type for configuration: {:?}",
                std::mem::discriminant(instance)),
        }),
    }
}

/// Extract a string from a Par (for map keys).
fn extract_string_from_par(par: &Par) -> Result<String, RawConfigError> {
    for expr in &par.exprs {
        if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
            return Ok(s.clone());
        }
    }
    Err(RawConfigError::InvalidMapKey {
        description: "Expected string key in map".to_string(),
    })
}

/// Parse a RawVectorDBConfig from a Par (the Rholang configuration map).
///
/// This function extracts `dimensions` (the only universal parameter) and
/// passes everything else through as generic configuration values.
///
/// # Arguments
///
/// * `par` - The Par representing the configuration map
///
/// # Returns
///
/// A `RawVectorDBConfig` with `dimensions` and all other parameters in `params`.
///
/// # Errors
///
/// Returns an error if:
/// - The Par is not a map
/// - The `dimensions` parameter is missing
/// - The `dimensions` parameter is not a valid positive integer
pub fn parse_raw_vectordb_config(par: &Par) -> Result<RawVectorDBConfig, RawConfigError> {
    // First, convert the entire Par to a RawConfigValue
    let config_value = par_to_raw_config_value(par)?;

    // Must be a map
    let config_map = match config_value {
        RawConfigValue::Map(m) => m,
        _ => return Err(RawConfigError::NotAMap),
    };

    // Extract dimensions (required)
    let dimensions = match config_map.get("dimensions") {
        Some(RawConfigValue::Int(i)) if *i > 0 => *i as usize,
        Some(RawConfigValue::String(s)) => {
            s.parse::<usize>().map_err(|_| RawConfigError::InvalidValue {
                parameter: "dimensions".to_string(),
                description: format!("Cannot parse '{}' as positive integer", s),
            })?
        }
        Some(other) => {
            return Err(RawConfigError::InvalidType {
                parameter: "dimensions".to_string(),
                expected: "positive integer".to_string(),
                actual: format!("{:?}", std::mem::discriminant(other)),
            });
        }
        None => {
            return Err(RawConfigError::MissingRequired {
                parameter: "dimensions".to_string(),
            });
        }
    };

    // Build params map (everything except dimensions)
    let mut params = HashMap::new();
    for (key, value) in config_map {
        if key != "dimensions" {
            params.insert(key, value);
        }
    }

    Ok(RawVectorDBConfig { dimensions, params })
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use models::rhoapi::{Expr, EMap, KeyValuePair};

    fn make_string_par(s: &str) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }],
            ..Default::default()
        }
    }

    fn make_int_par(i: i64) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(i)),
            }],
            ..Default::default()
        }
    }

    fn make_bool_par(b: bool) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GBool(b)),
            }],
            ..Default::default()
        }
    }

    fn make_map_par(kvs: Vec<(String, Par)>) -> Par {
        let kvs = kvs
            .into_iter()
            .map(|(k, v)| KeyValuePair {
                key: Some(make_string_par(&k)),
                value: Some(v),
            })
            .collect();
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EMapBody(EMap {
                    kvs,
                    locally_free: vec![],
                    connective_used: false,
                    remainder: None,
                })),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_raw_config_value_string() {
        let value = RawConfigValue::String("hello".to_string());
        assert_eq!(value.as_str(), Some("hello"));
        assert!(value.is_string());
        assert!(!value.is_int());
    }

    #[test]
    fn test_raw_config_value_int() {
        let value = RawConfigValue::Int(42);
        assert_eq!(value.as_int(), Some(42));
        assert_eq!(value.as_float(), Some(42.0));
        assert!(value.is_int());
    }

    #[test]
    fn test_raw_config_value_bool() {
        let value = RawConfigValue::Bool(true);
        assert_eq!(value.as_bool(), Some(true));
        assert!(value.is_bool());
    }

    #[test]
    fn test_raw_config_value_list() {
        let value = RawConfigValue::List(vec![
            RawConfigValue::String("a".to_string()),
            RawConfigValue::String("b".to_string()),
        ]);
        assert!(value.is_list());
        assert_eq!(value.as_list().map(|l| l.len()), Some(2));
    }

    #[test]
    fn test_raw_config_value_map() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), RawConfigValue::Int(123));
        let value = RawConfigValue::Map(map);
        assert!(value.is_map());
        assert_eq!(
            value.as_map().and_then(|m| m.get("key")).and_then(|v| v.as_int()),
            Some(123)
        );
    }

    #[test]
    fn test_raw_config_value_try_as_float() {
        assert_eq!(RawConfigValue::Float(0.5).try_as_float(), Some(0.5));
        assert_eq!(RawConfigValue::Int(42).try_as_float(), Some(42.0));
        assert_eq!(RawConfigValue::String("0.7".to_string()).try_as_float(), Some(0.7));
        assert_eq!(RawConfigValue::String("invalid".to_string()).try_as_float(), None);
        assert_eq!(RawConfigValue::Bool(true).try_as_float(), None);
    }

    #[test]
    fn test_raw_config_value_try_as_usize() {
        assert_eq!(RawConfigValue::Int(42).try_as_usize(), Some(42));
        assert_eq!(RawConfigValue::Int(-1).try_as_usize(), None);
        assert_eq!(RawConfigValue::String("100".to_string()).try_as_usize(), Some(100));
        assert_eq!(RawConfigValue::String("bad".to_string()).try_as_usize(), None);
    }

    #[test]
    fn test_raw_config_value_display() {
        assert_eq!(format!("{}", RawConfigValue::String("test".to_string())), "\"test\"");
        assert_eq!(format!("{}", RawConfigValue::Int(42)), "42");
        assert_eq!(format!("{}", RawConfigValue::Bool(true)), "true");
        assert_eq!(
            format!("{}", RawConfigValue::List(vec![RawConfigValue::Int(1), RawConfigValue::Int(2)])),
            "[1, 2]"
        );
    }

    #[test]
    fn test_raw_vectordb_config_new() {
        let config = RawVectorDBConfig::new(384);
        assert_eq!(config.dimensions, 384);
        assert!(config.params.is_empty());
    }

    #[test]
    fn test_raw_vectordb_config_with_params() {
        let mut params = HashMap::new();
        params.insert("metric".to_string(), RawConfigValue::String("cosine".to_string()));
        params.insert("threshold".to_string(), RawConfigValue::String("0.7".to_string()));

        let config = RawVectorDBConfig::with_params(384, params);
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.get_string("metric"), Some("cosine"));
        assert_eq!(config.get_float("threshold"), Some(0.7));
    }

    #[test]
    fn test_par_to_raw_config_value_string() {
        let par = make_string_par("hello");
        let value = par_to_raw_config_value(&par).expect("conversion failed");
        assert_eq!(value.as_str(), Some("hello"));
    }

    #[test]
    fn test_par_to_raw_config_value_int() {
        let par = make_int_par(42);
        let value = par_to_raw_config_value(&par).expect("conversion failed");
        assert_eq!(value.as_int(), Some(42));
    }

    #[test]
    fn test_par_to_raw_config_value_bool() {
        let par = make_bool_par(true);
        let value = par_to_raw_config_value(&par).expect("conversion failed");
        assert_eq!(value.as_bool(), Some(true));
    }

    #[test]
    fn test_par_to_raw_config_value_map() {
        let par = make_map_par(vec![
            ("key1".to_string(), make_string_par("value1")),
            ("key2".to_string(), make_int_par(42)),
        ]);
        let value = par_to_raw_config_value(&par).expect("conversion failed");
        let map = value.as_map().expect("should be map");
        assert_eq!(map.get("key1").and_then(|v| v.as_str()), Some("value1"));
        assert_eq!(map.get("key2").and_then(|v| v.as_int()), Some(42));
    }

    #[test]
    fn test_parse_raw_vectordb_config_basic() {
        let par = make_map_par(vec![
            ("dimensions".to_string(), make_int_par(384)),
            ("metric".to_string(), make_string_par("cosine")),
        ]);

        let config = parse_raw_vectordb_config(&par).expect("parse failed");
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.get_string("metric"), Some("cosine"));
        assert!(!config.contains_key("dimensions")); // dimensions removed from params
    }

    #[test]
    fn test_parse_raw_vectordb_config_with_index() {
        let par = make_map_par(vec![
            ("dimensions".to_string(), make_int_par(128)),
            ("metric".to_string(), make_string_par("euclidean")),
            ("threshold".to_string(), make_string_par("0.8")),
            ("index".to_string(), make_string_par("pre_normalize")),
        ]);

        let config = parse_raw_vectordb_config(&par).expect("parse failed");
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.get_string("metric"), Some("euclidean"));
        assert_eq!(config.get_float("threshold"), Some(0.8));
        assert_eq!(config.get_string("index"), Some("pre_normalize"));
    }

    #[test]
    fn test_parse_raw_vectordb_config_missing_dimensions() {
        let par = make_map_par(vec![
            ("metric".to_string(), make_string_par("cosine")),
        ]);

        let result = parse_raw_vectordb_config(&par);
        assert!(matches!(
            result,
            Err(RawConfigError::MissingRequired { parameter }) if parameter == "dimensions"
        ));
    }

    #[test]
    fn test_parse_raw_vectordb_config_nested_index() {
        // Test with nested map for index configuration
        let index_map = make_map_par(vec![
            ("type".to_string(), make_string_par("hnsw")),
            ("max_connections".to_string(), make_int_par(32)),
            ("ef_construction".to_string(), make_int_par(400)),
        ]);

        let par = make_map_par(vec![
            ("dimensions".to_string(), make_int_par(768)),
            ("index".to_string(), index_map),
        ]);

        let config = parse_raw_vectordb_config(&par).expect("parse failed");
        assert_eq!(config.dimensions, 768);

        let index = config.get("index").expect("index should exist");
        let index_map = index.as_map().expect("index should be map");
        assert_eq!(index_map.get("type").and_then(|v| v.as_str()), Some("hnsw"));
        assert_eq!(index_map.get("max_connections").and_then(|v| v.as_int()), Some(32));
        assert_eq!(index_map.get("ef_construction").and_then(|v| v.as_int()), Some(400));
    }

    #[test]
    fn test_raw_config_error_display() {
        let err = RawConfigError::MissingRequired { parameter: "dimensions".to_string() };
        assert!(format!("{}", err).contains("dimensions"));

        let err = RawConfigError::InvalidType {
            parameter: "threshold".to_string(),
            expected: "float".to_string(),
            actual: "string".to_string(),
        };
        assert!(format!("{}", err).contains("threshold"));
    }
}
