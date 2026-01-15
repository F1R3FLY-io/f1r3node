//! URN Configuration Parsing
//!
//! This module handles parsing URN strings into SpaceConfig, including
//! parameter extraction and legacy format support.

use super::urn::{InnerType, OuterType, Qualifier, is_valid_combination};
use super::super::types::{GasConfiguration, SpaceConfig, InnerCollectionType, OuterStorageType};

// =============================================================================
// URN Parsing with Parameters
// =============================================================================

/// Parameters for parametric inner types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InnerParams {
    None,
    PriorityQueue { priorities: usize },
    VectorDB {
        dimensions: usize,
        /// Backend name (e.g., "rho", "pinecone"). Default is "rho".
        backend: String,
    },
}

/// Parameters for parametric outer types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OuterParams {
    None,
    Array { size: usize, cyclic: bool },
}

/// Parse inner type with optional parameters from URN component.
///
/// Examples:
/// - "bag" → (InnerType::Bag, InnerParams::None)
/// - "priorityqueue(4)" → (InnerType::PriorityQueue, InnerParams::PriorityQueue { priorities: 4 })
/// - "vectordb(384)" → (InnerType::VectorDB, InnerParams::VectorDB { dimensions: 384 })
pub fn parse_inner_with_params(s: &str) -> Option<(InnerType, InnerParams)> {
    let inner_type = InnerType::from_str(s)?;

    let params = if let Some(start) = s.find('(') {
        if let Some(end) = s.find(')') {
            let param_str = &s[start + 1..end];
            match inner_type {
                InnerType::PriorityQueue => {
                    let priorities = param_str.trim().parse::<usize>().ok()?;
                    InnerParams::PriorityQueue { priorities }
                }
                InnerType::VectorDB => {
                    let dimensions = param_str.trim().parse::<usize>().ok()?;
                    // Backend defaults to "rho" for URN-based parsing
                    InnerParams::VectorDB {
                        dimensions,
                        backend: "rho".to_string(),
                    }
                }
                _ => InnerParams::None,
            }
        } else {
            return None; // Malformed: opening paren without closing
        }
    } else {
        // Default parameters for parametric types
        match inner_type {
            InnerType::PriorityQueue => InnerParams::PriorityQueue { priorities: 2 },
            InnerType::VectorDB => InnerParams::VectorDB {
                dimensions: 384,
                backend: "rho".to_string(),
            },
            _ => InnerParams::None,
        }
    };

    Some((inner_type, params))
}

/// Parse outer type with optional parameters from URN component.
///
/// Examples:
/// - "hashmap" → (OuterType::HashMap, OuterParams::None)
/// - "array(500,true)" → (OuterType::Array, OuterParams::Array { size: 500, cyclic: true })
pub fn parse_outer_with_params(s: &str) -> Option<(OuterType, OuterParams)> {
    let outer_type = OuterType::from_str(s)?;

    let params = if let Some(start) = s.find('(') {
        if let Some(end) = s.find(')') {
            let param_str = &s[start + 1..end];
            match outer_type {
                OuterType::Array => {
                    let parts: Vec<&str> = param_str.split(',').collect();
                    let size = parts.first()?.trim().parse::<usize>().ok()?;
                    let cyclic = parts.get(1).map(|s| s.trim() == "true").unwrap_or(false);
                    OuterParams::Array { size, cyclic }
                }
                _ => OuterParams::None,
            }
        } else {
            return None; // Malformed
        }
    } else {
        // Default parameters for array
        match outer_type {
            OuterType::Array => OuterParams::Array {
                size: 1000,
                cyclic: false,
            },
            _ => OuterParams::None,
        }
    };

    Some((outer_type, params))
}

/// Compute SpaceConfig from parsed URN components.
///
/// This is the core function that translates (InnerType, OuterType, Qualifier)
/// with their parameters into a fully configured SpaceConfig.
pub fn compute_config(
    inner: InnerType,
    inner_params: InnerParams,
    outer: OuterType,
    outer_params: OuterParams,
    qualifier: Qualifier,
) -> SpaceConfig {
    // Build outer storage type
    let outer_storage = match (outer, outer_params) {
        (OuterType::HashMap, _) => OuterStorageType::HashMap,
        (OuterType::PathMap, _) => OuterStorageType::PathMap,
        (OuterType::Array, OuterParams::Array { size, cyclic }) => OuterStorageType::Array {
            max_size: size,
            cyclic,
        },
        (OuterType::Array, _) => OuterStorageType::Array {
            max_size: 1000,
            cyclic: false,
        },
        (OuterType::Vector, _) => OuterStorageType::Vector,
        (OuterType::HashSet, _) => OuterStorageType::HashSet,
    };

    // Build inner collection type
    let data_collection = match (inner, inner_params.clone()) {
        (InnerType::Bag, _) => InnerCollectionType::Bag,
        (InnerType::Queue, _) => InnerCollectionType::Queue,
        (InnerType::Stack, _) => InnerCollectionType::Stack,
        (InnerType::Set, _) => InnerCollectionType::Set,
        (InnerType::Cell, _) => InnerCollectionType::Cell,
        (InnerType::PriorityQueue, InnerParams::PriorityQueue { priorities }) => {
            InnerCollectionType::PriorityQueue { priorities }
        }
        (InnerType::PriorityQueue, _) => InnerCollectionType::PriorityQueue { priorities: 2 },
        (InnerType::VectorDB, InnerParams::VectorDB { dimensions, backend }) => {
            InnerCollectionType::VectorDB { dimensions, backend }
        }
        (InnerType::VectorDB, _) => InnerCollectionType::VectorDB {
            dimensions: 384,
            backend: "rho".to_string(),
        },
    };

    // Continuation collection (same as data for most; Bag for VectorDB)
    let continuation_collection = match inner {
        InnerType::VectorDB => InnerCollectionType::Bag,
        _ => data_collection.clone(),
    };

    SpaceConfig {
        outer: outer_storage,
        data_collection,
        continuation_collection,
        qualifier: qualifier.to_space_qualifier(),
        theory: None,
        gas_config: GasConfiguration::default(),
    }
}

/// Parse a URN using computed pattern matching.
///
/// This is the new implementation that uses the InnerType/OuterType/Qualifier
/// enums for efficient parsing. Falls back to legacy parsing for short-form URNs.
///
/// Returns `None` if the URN is invalid or unsupported.
pub fn config_from_urn_computed(urn: &str) -> Option<SpaceConfig> {
    // Try extended format first: rho:space:{inner}:{outer}:{qualifier}
    if let Some(stripped) = urn.strip_prefix("rho:space:") {
        let parts: Vec<&str> = stripped.split(':').collect();

        if parts.len() >= 3 {
            // Parse components with optional parameters
            let (inner, inner_params) = parse_inner_with_params(parts[0])?;
            let (outer, outer_params) = parse_outer_with_params(parts[1])?;
            let qualifier = Qualifier::from_str(parts[2])?;

            // Validate combination
            if !is_valid_combination(inner, outer) {
                return None;
            }

            return Some(compute_config(inner, inner_params, outer, outer_params, qualifier));
        }
    }

    // Fall back to legacy parsing
    None
}

/// Get the SpaceConfig for a given URN.
///
/// This maps standard URNs to their corresponding configurations.
///
/// # Supported URN Formats
///
/// ## Short Format (legacy)
/// - `rho:space:HashMapBagSpace` - HashMap + Bag
/// - `rho:space:QueueSpace` - HashMap + Queue
/// - etc.
///
/// ## Extended Format (rho:space:{inner}:{outer}:{qualifier})
/// All valid combinations of:
/// - Inner: bag, queue, stack, set, cell, priorityqueue, vectordb
/// - Outer: hashmap, pathmap, array, vector, hashset
/// - Qualifier: default, temp, seq
///
/// Invalid combinations (rejected):
/// - VectorDB + PathMap/Array/HashSet (VectorDB needs O(1) lookup)
///
/// This extended format provides more granular control over space configuration.
pub fn config_from_urn(urn: &str) -> Option<SpaceConfig> {
    // Try computed pattern matching first (handles all extended format URNs)
    if let Some(config) = config_from_urn_computed(urn) {
        return Some(config);
    }

    // Fall back to legacy short-form URN handling
    match urn {
        // =====================================================================
        // Short Format (legacy) - rho:space:{SpaceType}
        // =====================================================================
        "rho:space:HashMapBagSpace" => Some(SpaceConfig::hashmap_bag()),
        "rho:space:PathMapSpace" => Some(SpaceConfig::pathmap()),
        "rho:space:QueueSpace" => Some(SpaceConfig::queue()),
        "rho:space:StackSpace" => Some(SpaceConfig::stack()),
        "rho:space:SetSpace" => Some(SpaceConfig::set()),
        "rho:space:CellSpace" => Some(SpaceConfig::cell()),
        "rho:space:VectorSpace" => Some(SpaceConfig::vector()),
        "rho:space:SeqSpace" => Some(SpaceConfig::seq()),
        "rho:space:TempSpace" => Some(SpaceConfig::temp()),

        // =====================================================================
        // Parameterized legacy spaces
        // =====================================================================
        _ if urn.starts_with("rho:space:ArraySpace") => {
            parse_array_config(urn)
        }
        _ if urn.starts_with("rho:space:PriorityQueueSpace") => {
            parse_priority_queue_config(urn)
        }
        _ if urn.starts_with("rho:space:VectorDBSpace") => {
            parse_vector_db_config(urn)
        }

        // =====================================================================
        // Legacy extended format with trailing parameters
        // (handled by computed for standard cases, but these have extra params)
        // =====================================================================
        _ if urn.starts_with("rho:space:priorityqueue:hashmap:default(") => {
            parse_extended_priority_queue_config(urn)
        }
        _ if urn.starts_with("rho:space:vectordb:hashmap:default(") => {
            parse_extended_vector_db_config(urn)
        }

        _ => None,
    }
}

// =============================================================================
// Legacy Parsing Helpers
// =============================================================================

/// Parse array space configuration from URN.
fn parse_array_config(urn: &str) -> Option<SpaceConfig> {
    // Default array config
    let mut max_size = 1000;
    let mut cyclic = false;

    // Try to parse parameters from URN
    if let Some(start) = urn.find('(') {
        if let Some(end) = urn.find(')') {
            let params = &urn[start + 1..end];
            let parts: Vec<&str> = params.split(',').collect();
            if let Some(size_str) = parts.first() {
                if let Ok(size) = size_str.trim().parse::<usize>() {
                    max_size = size;
                }
            }
            if let Some(cyclic_str) = parts.get(1) {
                cyclic = cyclic_str.trim() == "true";
            }
        }
    }

    Some(SpaceConfig::array(max_size, cyclic))
}

/// Parse priority queue configuration from URN.
fn parse_priority_queue_config(urn: &str) -> Option<SpaceConfig> {
    let mut priorities = 2; // Default to high/low

    if let Some(start) = urn.find('(') {
        if let Some(end) = urn.find(')') {
            let param = &urn[start + 1..end];
            if let Ok(p) = param.trim().parse::<usize>() {
                priorities = p;
            }
        }
    }

    Some(SpaceConfig::priority_queue(priorities))
}

/// Parse vector DB configuration from URN.
fn parse_vector_db_config(urn: &str) -> Option<SpaceConfig> {
    // VectorDB requires the vectordb feature
    #[cfg(not(feature = "vectordb"))]
    {
        tracing::warn!(
            "VectorDB feature not enabled. Recompile with --features vectordb. URN: {}",
            urn
        );
        return None;
    }

    #[cfg(feature = "vectordb")]
    {
        let mut dimensions = 384; // Default to common embedding size

        if let Some(start) = urn.find('(') {
            if let Some(end) = urn.find(')') {
                let param = &urn[start + 1..end];
                if let Ok(d) = param.trim().parse::<usize>() {
                    dimensions = d;
                }
            }
        }

        Some(SpaceConfig::vector_db(dimensions))
    }
}

/// Parse extended priority queue configuration.
/// Format: rho:space:priorityqueue:hashmap:default(n)
fn parse_extended_priority_queue_config(urn: &str) -> Option<SpaceConfig> {
    let mut priorities = 2; // Default

    if let Some(start) = urn.find('(') {
        if let Some(end) = urn.find(')') {
            let param = &urn[start + 1..end];
            if let Ok(p) = param.trim().parse::<usize>() {
                priorities = p;
            }
        }
    }

    Some(SpaceConfig::priority_queue(priorities))
}

/// Parse extended vector DB configuration.
/// Format: rho:space:vectordb:hashmap:default(dims)
fn parse_extended_vector_db_config(urn: &str) -> Option<SpaceConfig> {
    // VectorDB requires the vectordb feature
    #[cfg(not(feature = "vectordb"))]
    {
        tracing::warn!(
            "VectorDB feature not enabled. Recompile with --features vectordb. URN: {}",
            urn
        );
        return None;
    }

    #[cfg(feature = "vectordb")]
    {
        let mut dimensions = 384; // Default to common embedding size

        if let Some(start) = urn.find('(') {
            if let Some(end) = urn.find(')') {
                let param = &urn[start + 1..end];
                if let Ok(d) = param.trim().parse::<usize>() {
                    dimensions = d;
                }
            }
        }

        Some(SpaceConfig::vector_db(dimensions))
    }
}

/// Parse inner collection type from string (used by legacy extended format).
pub fn parse_inner_collection_type(s: &str) -> Option<InnerCollectionType> {
    if s.starts_with("priorityqueue(") {
        let start = s.find('(')?;
        let end = s.find(')')?;
        let priorities = s[start + 1..end].trim().parse::<usize>().ok()?;
        Some(InnerCollectionType::PriorityQueue { priorities })
    } else if s.starts_with("vectordb(") {
        let start = s.find('(')?;
        let end = s.find(')')?;
        let dimensions = s[start + 1..end].trim().parse::<usize>().ok()?;
        // Backend defaults to "rho" for URN-based parsing
        Some(InnerCollectionType::VectorDB {
            dimensions,
            backend: "rho".to_string(),
        })
    } else {
        match s {
            "bag" => Some(InnerCollectionType::Bag),
            "queue" => Some(InnerCollectionType::Queue),
            "stack" => Some(InnerCollectionType::Stack),
            "set" => Some(InnerCollectionType::Set),
            "cell" => Some(InnerCollectionType::Cell),
            "priorityqueue" => Some(InnerCollectionType::PriorityQueue { priorities: 2 }),
            "vectordb" => Some(InnerCollectionType::VectorDB {
                dimensions: 384,
                backend: "rho".to_string(),
            }),
            _ => None,
        }
    }
}

/// Parse outer storage type from string (used by legacy extended format).
pub fn parse_outer_storage_type(s: &str) -> Option<OuterStorageType> {
    match s {
        "hashmap" => Some(OuterStorageType::HashMap),
        "pathmap" => Some(OuterStorageType::PathMap),
        "vector" => Some(OuterStorageType::Vector),
        "hashset" => Some(OuterStorageType::HashSet),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_urn_basic() {
        let config = config_from_urn("rho:space:HashMapBagSpace").expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);
    }

    #[test]
    fn test_config_from_urn_queue() {
        let config = config_from_urn("rho:space:QueueSpace").expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Queue);
    }

    #[test]
    fn test_config_from_urn_array() {
        let config = config_from_urn("rho:space:ArraySpace(500,true)").expect("Should parse");
        match config.outer {
            OuterStorageType::Array { max_size, cyclic } => {
                assert_eq!(max_size, 500);
                assert!(cyclic);
            }
            _ => panic!("Expected Array outer type"),
        }
    }

    #[test]
    fn test_config_from_urn_priority_queue() {
        let config = config_from_urn("rho:space:PriorityQueueSpace(4)").expect("Should parse");
        match config.data_collection {
            InnerCollectionType::PriorityQueue { priorities } => {
                assert_eq!(priorities, 4);
            }
            _ => panic!("Expected PriorityQueue collection type"),
        }
    }

    #[test]
    fn test_parse_inner_with_params_simple() {
        let (inner, params) = parse_inner_with_params("bag").expect("Should parse");
        assert_eq!(inner, InnerType::Bag);
        assert_eq!(params, InnerParams::None);

        let (inner, params) = parse_inner_with_params("queue").expect("Should parse");
        assert_eq!(inner, InnerType::Queue);
        assert_eq!(params, InnerParams::None);
    }

    #[test]
    fn test_parse_inner_with_params_priorityqueue() {
        let (inner, params) = parse_inner_with_params("priorityqueue").expect("Should parse");
        assert_eq!(inner, InnerType::PriorityQueue);
        assert_eq!(params, InnerParams::PriorityQueue { priorities: 2 }); // Default

        let (inner, params) = parse_inner_with_params("priorityqueue(8)").expect("Should parse");
        assert_eq!(inner, InnerType::PriorityQueue);
        assert_eq!(params, InnerParams::PriorityQueue { priorities: 8 });
    }

    #[test]
    fn test_parse_inner_with_params_vectordb() {
        let (inner, params) = parse_inner_with_params("vectordb").expect("Should parse");
        assert_eq!(inner, InnerType::VectorDB);
        assert_eq!(
            params,
            InnerParams::VectorDB {
                dimensions: 384,
                backend: "rho".to_string(),
            }
        ); // Default

        let (inner, params) = parse_inner_with_params("vectordb(128)").expect("Should parse");
        assert_eq!(inner, InnerType::VectorDB);
        assert_eq!(
            params,
            InnerParams::VectorDB {
                dimensions: 128,
                backend: "rho".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_outer_with_params_simple() {
        let (outer, params) = parse_outer_with_params("hashmap").expect("Should parse");
        assert_eq!(outer, OuterType::HashMap);
        assert_eq!(params, OuterParams::None);

        let (outer, params) = parse_outer_with_params("pathmap").expect("Should parse");
        assert_eq!(outer, OuterType::PathMap);
        assert_eq!(params, OuterParams::None);
    }

    #[test]
    fn test_parse_outer_with_params_array() {
        let (outer, params) = parse_outer_with_params("array").expect("Should parse");
        assert_eq!(outer, OuterType::Array);
        assert_eq!(params, OuterParams::Array { size: 1000, cyclic: false }); // Default

        let (outer, params) = parse_outer_with_params("array(500,true)").expect("Should parse");
        assert_eq!(outer, OuterType::Array);
        assert_eq!(params, OuterParams::Array { size: 500, cyclic: true });

        let (outer, params) = parse_outer_with_params("array(250,false)").expect("Should parse");
        assert_eq!(outer, OuterType::Array);
        assert_eq!(params, OuterParams::Array { size: 250, cyclic: false });
    }

    #[test]
    fn test_compute_config_basic() {
        let config = compute_config(
            InnerType::Bag,
            InnerParams::None,
            OuterType::HashMap,
            OuterParams::None,
            Qualifier::Default,
        );
        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);
        assert_eq!(config.continuation_collection, InnerCollectionType::Bag);
    }

    #[test]
    fn test_compute_config_array_with_params() {
        let config = compute_config(
            InnerType::Queue,
            InnerParams::None,
            OuterType::Array,
            OuterParams::Array { size: 500, cyclic: true },
            Qualifier::Temp,
        );
        match config.outer {
            OuterStorageType::Array { max_size, cyclic } => {
                assert_eq!(max_size, 500);
                assert!(cyclic);
            }
            _ => panic!("Expected Array outer type"),
        }
        assert_eq!(config.data_collection, InnerCollectionType::Queue);
    }

    #[test]
    fn test_compute_config_vectordb() {
        let config = compute_config(
            InnerType::VectorDB,
            InnerParams::VectorDB {
                dimensions: 128,
                backend: "rho".to_string(),
            },
            OuterType::HashMap,
            OuterParams::None,
            Qualifier::Default,
        );
        match config.data_collection {
            InnerCollectionType::VectorDB { dimensions, backend } => {
                assert_eq!(dimensions, 128);
                assert_eq!(backend, "rho");
            }
            _ => panic!("Expected VectorDB data collection"),
        }
        // VectorDB uses Bag for continuations
        assert_eq!(config.continuation_collection, InnerCollectionType::Bag);
    }

    #[test]
    fn test_config_from_urn_computed_basic() {
        let config = config_from_urn_computed("rho:space:bag:hashmap:default").expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);
    }

    #[test]
    fn test_config_from_urn_computed_with_params() {
        let config = config_from_urn_computed("rho:space:priorityqueue(4):hashmap:default").expect("Should parse");
        match config.data_collection {
            InnerCollectionType::PriorityQueue { priorities } => {
                assert_eq!(priorities, 4);
            }
            _ => panic!("Expected PriorityQueue"),
        }

        let config = config_from_urn_computed("rho:space:stack:array(256,true):temp").expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Stack);
        match config.outer {
            OuterStorageType::Array { max_size, cyclic } => {
                assert_eq!(max_size, 256);
                assert!(cyclic);
            }
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_config_from_urn_computed_invalid() {
        // Invalid inner type
        assert!(config_from_urn_computed("rho:space:invalid:hashmap:default").is_none());
        // Invalid outer type
        assert!(config_from_urn_computed("rho:space:bag:invalid:default").is_none());
        // Invalid qualifier
        assert!(config_from_urn_computed("rho:space:bag:hashmap:invalid").is_none());
        // Invalid combination
        assert!(config_from_urn_computed("rho:space:vectordb:pathmap:default").is_none());
    }

    #[test]
    fn test_config_from_urn_computed_legacy_fallback() {
        // Legacy short-form URNs should return None (not handled by computed)
        assert!(config_from_urn_computed("rho:space:HashMapBagSpace").is_none());
        assert!(config_from_urn_computed("rho:space:QueueSpace").is_none());
    }
}
