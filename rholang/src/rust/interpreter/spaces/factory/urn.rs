//! URN Parsing Types and Utilities
//!
//! This module defines the core types for parsing space URNs:
//! - InnerType, OuterType, Qualifier enums for pattern matching
//! - Combination validation
//! - URN iteration and byte name mapping

use std::fmt;

use super::super::types::{SpaceQualifier, InnerCollectionType, OuterStorageType, SpaceConfig};

// =============================================================================
// URN Parsing Enums for Pattern Matching
// =============================================================================
//
// These enums provide a simple, flat representation of URN components for
// efficient pattern-matching based parsing. Unlike InnerCollectionType/
// OuterStorageType/SpaceQualifier which include parameters, these enums are
// parameter-free for clean parsing and exhaustive matching.
//
// Usage:
//   1. Parse URN string into (InnerType, OuterType, Qualifier) tuple
//   2. Validate combination using is_valid_combination()
//   3. Compute SpaceConfig using compute_config()
// =============================================================================

/// Inner collection type for URN parsing.
///
/// Maps to `InnerCollectionType` but without parameters.
/// Parameters (like priority count or dimensions) are parsed separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InnerType {
    Bag,
    Queue,
    Stack,
    Set,
    Cell,
    PriorityQueue,
    VectorDB,
}

impl InnerType {
    /// Parse inner type from URN component string.
    ///
    /// Handles both simple names ("bag") and parametric prefixes ("priorityqueue(4)").
    pub fn from_str(s: &str) -> Option<Self> {
        // Strip any parameters for matching
        let base = if let Some(idx) = s.find('(') {
            &s[..idx]
        } else {
            s
        };

        match base {
            "bag" => Some(Self::Bag),
            "queue" => Some(Self::Queue),
            "stack" => Some(Self::Stack),
            "set" => Some(Self::Set),
            "cell" => Some(Self::Cell),
            "priorityqueue" => Some(Self::PriorityQueue),
            "vectordb" => Some(Self::VectorDB),
            _ => None,
        }
    }

    /// Get the string representation for URN construction.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bag => "bag",
            Self::Queue => "queue",
            Self::Stack => "stack",
            Self::Set => "set",
            Self::Cell => "cell",
            Self::PriorityQueue => "priorityqueue",
            Self::VectorDB => "vectordb",
        }
    }

    /// All inner types for iteration.
    pub const ALL: [Self; 7] = [
        Self::Bag,
        Self::Queue,
        Self::Stack,
        Self::Set,
        Self::Cell,
        Self::PriorityQueue,
        Self::VectorDB,
    ];
}

impl fmt::Display for InnerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Outer storage type for URN parsing.
///
/// Maps to `OuterStorageType` but without parameters.
/// Parameters (like array size) are parsed separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OuterType {
    HashMap,
    PathMap,
    Array,
    Vector,
    HashSet,
}

impl OuterType {
    /// Parse outer type from URN component string.
    ///
    /// Handles both simple names ("hashmap") and parametric prefixes ("array(100,true)").
    pub fn from_str(s: &str) -> Option<Self> {
        // Strip any parameters for matching
        let base = if let Some(idx) = s.find('(') {
            &s[..idx]
        } else {
            s
        };

        match base {
            "hashmap" => Some(Self::HashMap),
            "pathmap" => Some(Self::PathMap),
            "array" => Some(Self::Array),
            "vector" => Some(Self::Vector),
            "hashset" => Some(Self::HashSet),
            _ => None,
        }
    }

    /// Get the string representation for URN construction.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HashMap => "hashmap",
            Self::PathMap => "pathmap",
            Self::Array => "array",
            Self::Vector => "vector",
            Self::HashSet => "hashset",
        }
    }

    /// All outer types for iteration.
    pub const ALL: [Self; 5] = [
        Self::HashMap,
        Self::PathMap,
        Self::Array,
        Self::Vector,
        Self::HashSet,
    ];
}

impl fmt::Display for OuterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Qualifier type for URN parsing.
///
/// Maps directly to `SpaceQualifier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qualifier {
    Default,
    Temp,
    Seq,
}

impl Qualifier {
    /// Parse qualifier from URN component string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "temp" => Some(Self::Temp),
            "seq" => Some(Self::Seq),
            _ => None,
        }
    }

    /// Get the string representation for URN construction.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Temp => "temp",
            Self::Seq => "seq",
        }
    }

    /// Convert to SpaceQualifier.
    pub fn to_space_qualifier(&self) -> SpaceQualifier {
        match self {
            Self::Default => SpaceQualifier::Default,
            Self::Temp => SpaceQualifier::Temp,
            Self::Seq => SpaceQualifier::Seq,
        }
    }

    /// All qualifiers for iteration.
    pub const ALL: [Self; 3] = [Self::Default, Self::Temp, Self::Seq];
}

impl fmt::Display for Qualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Combination Validation
// =============================================================================

/// Check if an (inner, outer) combination is valid.
///
/// Invalid combinations:
/// - VectorDB + PathMap (VectorDB needs O(1) lookup, PathMap has different semantics)
/// - VectorDB + Array (VectorDB is incompatible with fixed-size storage)
/// - VectorDB + HashSet (VectorDB requires full storage, not presence-only)
///
/// All other combinations are valid.
pub fn is_valid_combination(inner: InnerType, outer: OuterType) -> bool {
    match (inner, outer) {
        // VectorDB only works with HashMap and Vector
        (InnerType::VectorDB, OuterType::HashMap) => true,
        (InnerType::VectorDB, OuterType::Vector) => true,
        (InnerType::VectorDB, _) => false,
        // All other combinations are valid
        _ => true,
    }
}

// =============================================================================
// URN Iteration
// =============================================================================

/// Iterator over all valid URN combinations.
///
/// Yields URNs in the format `rho:space:{inner}:{outer}:{qualifier}`.
/// Skips invalid combinations (e.g., vectordb:pathmap:*).
///
/// # Example
/// ```ignore
/// for urn in all_valid_urns() {
///     println!("{}", urn);  // e.g., "rho:space:bag:hashmap:default"
/// }
/// ```
pub fn all_valid_urns() -> impl Iterator<Item = String> {
    InnerType::ALL.iter().flat_map(|&inner| {
        OuterType::ALL
            .iter()
            .filter(move |&&outer| is_valid_combination(inner, outer))
            .flat_map(move |&outer| {
                Qualifier::ALL.iter().map(move |&qualifier| {
                    format!(
                        "rho:space:{}:{}:{}",
                        inner.as_str(),
                        outer.as_str(),
                        qualifier.as_str()
                    )
                })
            })
    })
}

/// Count of all valid URN combinations.
///
/// Calculation:
/// - 7 inner types × 5 outer types = 35 base combinations
/// - VectorDB has 3 invalid outer types (PathMap, Array, HashSet) = 3 invalid
/// - Valid combinations = 35 - 3 = 32
/// - Each combination × 3 qualifiers = 32 × 3 = 96 valid URNs
pub fn valid_urn_count() -> usize {
    all_valid_urns().count()
}

// =============================================================================
// Byte Name Mapping
// =============================================================================

/// Base byte for space factory URNs.
///
/// Bytes 0-24 are reserved for standard system processes (stdout, crypto, etc.).
/// Byte 150 is reserved for vector operations.
/// Starting at 25 provides space for all 96 valid URN combinations (bytes 25-120).
const SPACE_FACTORY_BASE_BYTE: u8 = 25;

/// Get the deterministic byte name for a space factory URN.
///
/// Base byte is 25 (avoiding bytes 0-24 for standard system processes),
/// incrementing for each valid combination.
///
/// Returns `None` if the URN is invalid or doesn't match the expected format.
///
/// # Note
/// This function is deterministic - the same URN always produces the same byte.
/// With 96 valid combinations starting at 25, bytes range from 25-120.
pub fn urn_to_byte_name(urn: &str) -> Option<u8> {
    let stripped = urn.strip_prefix("rho:space:")?;
    let parts: Vec<&str> = stripped.split(':').collect();
    if parts.len() < 3 {
        return None;
    }

    let inner = InnerType::from_str(parts[0])?;
    let outer = OuterType::from_str(parts[1])?;
    let qualifier = Qualifier::from_str(parts[2])?;

    if !is_valid_combination(inner, outer) {
        return None;
    }

    // Sequential enumeration to compute unique byte starting at SPACE_FACTORY_BASE_BYTE
    let mut byte: u8 = SPACE_FACTORY_BASE_BYTE;
    for &i in &InnerType::ALL {
        for &o in &OuterType::ALL {
            if !is_valid_combination(i, o) {
                continue;
            }
            for &q in &Qualifier::ALL {
                if i == inner && o == outer && q == qualifier {
                    return Some(byte);
                }
                byte = byte.wrapping_add(1);
            }
        }
    }
    None
}

/// Get the URN for a given byte name.
///
/// This is the inverse of `urn_to_byte_name`.
/// Returns `None` if the byte doesn't correspond to a valid URN.
///
/// Note: With 96 valid URNs starting at byte 25, bytes range from 25-120.
pub fn byte_name_to_urn(byte: u8) -> Option<String> {
    // Calculate index from byte, accounting for base byte offset
    let total_urns = 96; // Precomputed: 32 inner×outer combos × 3 qualifiers
    let max_byte = SPACE_FACTORY_BASE_BYTE + total_urns as u8 - 1; // 120

    if byte < SPACE_FACTORY_BASE_BYTE || byte > max_byte {
        return None; // byte is outside the valid space factory range (25-120)
    }

    let target_idx = (byte - SPACE_FACTORY_BASE_BYTE) as usize;
    all_valid_urns().nth(target_idx)
}

// =============================================================================
// URN Helpers
// =============================================================================

/// Get the URN for a given SpaceConfig as a `Cow<'static, str>`.
///
/// This is the inverse of `config_from_urn`.
/// Returns a borrowed `&'static str` for common configurations, avoiding allocation.
/// Returns an owned String only for configurations requiring dynamic values.
///
/// This is the preferred method for performance-sensitive code.
pub fn urn_from_config_cow(config: &SpaceConfig) -> std::borrow::Cow<'static, str> {
    use std::borrow::Cow;

    let qualifier_str = match config.qualifier {
        SpaceQualifier::Default => "default",
        SpaceQualifier::Temp => "temp",
        SpaceQualifier::Seq => "seq",
    };

    // For legacy compatibility, return short-form URNs for common configurations
    // These return borrowed static strings - no allocation!
    match (&config.outer, &config.data_collection, config.qualifier) {
        // Legacy short-form URNs for HashMap + Bag
        (OuterStorageType::HashMap, InnerCollectionType::Bag, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:HashMapBagSpace");
        }
        (OuterStorageType::HashMap, InnerCollectionType::Bag, SpaceQualifier::Temp) => {
            return Cow::Borrowed("rho:space:TempSpace");
        }
        (OuterStorageType::PathMap, InnerCollectionType::Bag, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:PathMapSpace");
        }
        (OuterStorageType::HashSet, InnerCollectionType::Set, SpaceQualifier::Seq) => {
            return Cow::Borrowed("rho:space:SeqSpace");
        }
        // Legacy short-form URNs for HashMap + other inner types
        (OuterStorageType::HashMap, InnerCollectionType::Queue, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:QueueSpace");
        }
        (OuterStorageType::HashMap, InnerCollectionType::Stack, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:StackSpace");
        }
        (OuterStorageType::HashMap, InnerCollectionType::Set, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:SetSpace");
        }
        (OuterStorageType::HashMap, InnerCollectionType::Cell, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:CellSpace");
        }
        (OuterStorageType::Vector, InnerCollectionType::Bag, SpaceQualifier::Default) => {
            return Cow::Borrowed("rho:space:VectorSpace");
        }
        _ => {}
    }

    // Extended format URN for all other combinations
    let outer_str = match &config.outer {
        OuterStorageType::HashMap => "hashmap",
        OuterStorageType::PathMap => "pathmap",
        OuterStorageType::Array { max_size, cyclic } => {
            return Cow::Owned(format!(
                "rho:space:{}:array({},{}):{}",
                inner_collection_to_cow(&config.data_collection),
                max_size,
                cyclic,
                qualifier_str
            ));
        }
        OuterStorageType::Vector => "vector",
        OuterStorageType::HashSet => "hashset",
    };

    let inner_str = inner_collection_to_cow(&config.data_collection);

    Cow::Owned(format!("rho:space:{}:{}:{}", inner_str, outer_str, qualifier_str))
}

/// Get the URN for a given SpaceConfig.
///
/// This is the inverse of `config_from_urn`.
/// Returns the extended format URN: `rho:space:{inner}:{outer}:{qualifier}`
///
/// For performance-sensitive code, consider using `urn_from_config_cow` instead,
/// which avoids allocation for common configurations.
pub fn urn_from_config(config: &SpaceConfig) -> String {
    urn_from_config_cow(config).into_owned()
}

/// Convert an InnerCollectionType to its string representation for URN construction.
///
/// Uses `Cow` to avoid allocation for common types (Bag, Queue, Stack, Set, Cell).
/// Only parametric types (PriorityQueue, VectorDB) require allocation.
pub fn inner_collection_to_cow(collection: &InnerCollectionType) -> std::borrow::Cow<'static, str> {
    use std::borrow::Cow;
    match collection {
        InnerCollectionType::Bag => Cow::Borrowed("bag"),
        InnerCollectionType::Queue => Cow::Borrowed("queue"),
        InnerCollectionType::Stack => Cow::Borrowed("stack"),
        InnerCollectionType::Set => Cow::Borrowed("set"),
        InnerCollectionType::Cell => Cow::Borrowed("cell"),
        InnerCollectionType::PriorityQueue { priorities } => {
            Cow::Owned(format!("priorityqueue({})", priorities))
        }
        InnerCollectionType::VectorDB { dimensions, backend: _ } => {
            // Note: backend is not included in URN - all URN-based spaces use default "rho" backend
            Cow::Owned(format!("vectordb({})", dimensions))
        }
    }
}

/// Convert an InnerCollectionType to its string representation for URN construction.
///
/// This version always returns an owned String for backward compatibility.
/// Prefer `inner_collection_to_cow` for better performance when possible.
pub fn inner_collection_to_str(collection: &InnerCollectionType) -> String {
    inner_collection_to_cow(collection).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_type_from_str() {
        assert_eq!(InnerType::from_str("bag"), Some(InnerType::Bag));
        assert_eq!(InnerType::from_str("queue"), Some(InnerType::Queue));
        assert_eq!(InnerType::from_str("stack"), Some(InnerType::Stack));
        assert_eq!(InnerType::from_str("set"), Some(InnerType::Set));
        assert_eq!(InnerType::from_str("cell"), Some(InnerType::Cell));
        assert_eq!(InnerType::from_str("priorityqueue"), Some(InnerType::PriorityQueue));
        assert_eq!(InnerType::from_str("vectordb"), Some(InnerType::VectorDB));
        assert_eq!(InnerType::from_str("invalid"), None);
    }

    #[test]
    fn test_inner_type_from_str_with_params() {
        assert_eq!(InnerType::from_str("priorityqueue(4)"), Some(InnerType::PriorityQueue));
        assert_eq!(InnerType::from_str("vectordb(128)"), Some(InnerType::VectorDB));
        assert_eq!(InnerType::from_str("bag()"), Some(InnerType::Bag));
    }

    #[test]
    fn test_outer_type_from_str() {
        assert_eq!(OuterType::from_str("hashmap"), Some(OuterType::HashMap));
        assert_eq!(OuterType::from_str("pathmap"), Some(OuterType::PathMap));
        assert_eq!(OuterType::from_str("array"), Some(OuterType::Array));
        assert_eq!(OuterType::from_str("vector"), Some(OuterType::Vector));
        assert_eq!(OuterType::from_str("hashset"), Some(OuterType::HashSet));
        assert_eq!(OuterType::from_str("invalid"), None);
    }

    #[test]
    fn test_qualifier_from_str() {
        assert_eq!(Qualifier::from_str("default"), Some(Qualifier::Default));
        assert_eq!(Qualifier::from_str("temp"), Some(Qualifier::Temp));
        assert_eq!(Qualifier::from_str("seq"), Some(Qualifier::Seq));
        assert_eq!(Qualifier::from_str("invalid"), None);
    }

    #[test]
    fn test_is_valid_combination_vectordb() {
        assert!(is_valid_combination(InnerType::VectorDB, OuterType::HashMap));
        assert!(is_valid_combination(InnerType::VectorDB, OuterType::Vector));
        assert!(!is_valid_combination(InnerType::VectorDB, OuterType::PathMap));
        assert!(!is_valid_combination(InnerType::VectorDB, OuterType::Array));
        assert!(!is_valid_combination(InnerType::VectorDB, OuterType::HashSet));
    }

    #[test]
    fn test_all_valid_urns_count() {
        let count = valid_urn_count();
        assert_eq!(count, 96, "Expected 96 valid URNs, got {}", count);
    }

    #[test]
    fn test_urn_to_byte_name_basic() {
        let byte = urn_to_byte_name("rho:space:bag:hashmap:default");
        assert_eq!(byte, Some(25));

        let byte = urn_to_byte_name("rho:space:bag:hashmap:temp");
        assert_eq!(byte, Some(26));

        let byte = urn_to_byte_name("rho:space:bag:hashmap:seq");
        assert_eq!(byte, Some(27));
    }

    #[test]
    fn test_byte_name_to_urn_roundtrip() {
        for urn in all_valid_urns() {
            let byte = urn_to_byte_name(&urn).expect(&format!("Should get byte for {}", urn));
            let recovered = byte_name_to_urn(byte).expect(&format!("Should recover URN from byte {}", byte));
            assert_eq!(urn, recovered, "Roundtrip failed for {}", urn);
        }
    }

    #[test]
    fn test_urn_from_config_cow_returns_borrowed_for_common_configs() {
        use std::borrow::Cow;

        // Common configs should return Borrowed (no allocation)
        let config = SpaceConfig::hashmap_bag();
        let urn = urn_from_config_cow(&config);
        assert!(matches!(urn, Cow::Borrowed(_)), "HashMapBagSpace should be borrowed");
        assert_eq!(urn.as_ref(), "rho:space:HashMapBagSpace");

        let config = SpaceConfig::temp();
        let urn = urn_from_config_cow(&config);
        assert!(matches!(urn, Cow::Borrowed(_)), "TempSpace should be borrowed");
        assert_eq!(urn.as_ref(), "rho:space:TempSpace");

        let config = SpaceConfig::queue();
        let urn = urn_from_config_cow(&config);
        assert!(matches!(urn, Cow::Borrowed(_)), "QueueSpace should be borrowed");
        assert_eq!(urn.as_ref(), "rho:space:QueueSpace");

        // Parametric configs require allocation (Owned)
        let config = SpaceConfig::priority_queue(4);
        let urn = urn_from_config_cow(&config);
        assert!(matches!(urn, Cow::Owned(_)), "PriorityQueue should be owned");
        assert!(urn.as_ref().contains("priorityqueue(4)"));
    }

    #[test]
    fn test_inner_collection_to_cow_returns_borrowed_for_simple_types() {
        use std::borrow::Cow;

        // Simple types should return Borrowed
        assert!(matches!(inner_collection_to_cow(&InnerCollectionType::Bag), Cow::Borrowed(_)));
        assert!(matches!(inner_collection_to_cow(&InnerCollectionType::Queue), Cow::Borrowed(_)));
        assert!(matches!(inner_collection_to_cow(&InnerCollectionType::Stack), Cow::Borrowed(_)));
        assert!(matches!(inner_collection_to_cow(&InnerCollectionType::Set), Cow::Borrowed(_)));
        assert!(matches!(inner_collection_to_cow(&InnerCollectionType::Cell), Cow::Borrowed(_)));

        // Parametric types should return Owned
        assert!(matches!(
            inner_collection_to_cow(&InnerCollectionType::PriorityQueue { priorities: 4 }),
            Cow::Owned(_)
        ));
        assert!(matches!(
            inner_collection_to_cow(&InnerCollectionType::VectorDB { dimensions: 128, backend: "rho".to_string() }),
            Cow::Owned(_)
        ));
    }
}
