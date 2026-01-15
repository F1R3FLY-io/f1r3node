//! MeTTaIL Theory Integration
//!
//! This module provides types and traits for loading type theories from
//! various sources, enabling typed tuple spaces with MeTTaIL integration.

use std::fmt;
use std::sync::Arc;

use super::super::errors::SpaceError;
use super::super::types::{BoxedTheory, SpaceConfig};
use super::config::config_from_urn;

// =============================================================================
// Theory Specification
// =============================================================================

/// Specification of where to load a theory from.
///
/// This is parsed from URN extensions like `[theory=Nat]` or `[theory=file:nat.metta]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TheorySpec {
    /// A built-in theory referenced by name.
    /// Example: `[theory=Nat]`
    Builtin(String),

    /// Load theory from a MeTTaIL file.
    /// Example: `[theory=mettail:types/nat.metta]`
    MeTTaILFile(String),

    /// Parse inline MeTTaIL code.
    /// Example: `[theory=inline:(: Nat Type)]`
    InlineMeTTaIL(String),

    /// A URI reference to an external theory.
    /// Example: `[theory=uri:https://example.com/theories/nat.metta]`
    Uri(String),
}

impl fmt::Display for TheorySpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TheorySpec::Builtin(name) => write!(f, "{}", name),
            TheorySpec::MeTTaILFile(path) => write!(f, "mettail:{}", path),
            TheorySpec::InlineMeTTaIL(code) => write!(f, "inline:{}", code),
            TheorySpec::Uri(uri) => write!(f, "uri:{}", uri),
        }
    }
}

impl TheorySpec {
    /// Parse a theory specification from a string.
    ///
    /// # Examples
    /// - `"Nat"` → `TheorySpec::Builtin("Nat")`
    /// - `"mettail:types/nat.metta"` → `TheorySpec::MeTTaILFile("types/nat.metta")`
    /// - `"inline:(: Nat Type)"` → `TheorySpec::InlineMeTTaIL("(: Nat Type)")`
    pub fn parse(s: &str) -> Self {
        if let Some(path) = s.strip_prefix("mettail:") {
            TheorySpec::MeTTaILFile(path.to_string())
        } else if let Some(code) = s.strip_prefix("inline:") {
            TheorySpec::InlineMeTTaIL(code.to_string())
        } else if let Some(uri) = s.strip_prefix("uri:") {
            TheorySpec::Uri(uri.to_string())
        } else {
            TheorySpec::Builtin(s.to_string())
        }
    }
}

// =============================================================================
// Theory Loader Trait
// =============================================================================

/// Trait for loading theories from various sources.
///
/// Implementations of this trait handle the actual loading and parsing of
/// theories from different sources (built-in, files, inline, URIs).
///
/// # MeTTaIL Integration
///
/// The primary implementation should integrate with MeTTaIL to parse and
/// compile type theories and contracts. This trait provides the interface
/// for that integration.
///
/// # Example Implementation
///
/// ```ignore
/// struct MeTTaILTheoryLoader {
///     mettail: MeTTaIL,  // The MeTTaIL compiler instance
/// }
///
/// impl TheoryLoader for MeTTaILTheoryLoader {
///     fn load(&self, spec: &TheorySpec) -> Result<BoxedTheory, SpaceError> {
///         match spec {
///             TheorySpec::MeTTaILFile(path) => {
///                 let source = std::fs::read_to_string(path)?;
///                 let theory = self.mettail.parse_theory(&source)?;
///                 Ok(Box::new(theory))
///             }
///             TheorySpec::InlineMeTTaIL(code) => {
///                 let theory = self.mettail.parse_theory(code)?;
///                 Ok(Box::new(theory))
///             }
///             // ... other cases
///         }
///     }
/// }
/// ```
pub trait TheoryLoader: Send + Sync {
    /// Load a theory from the given specification.
    ///
    /// # Arguments
    /// - `spec`: The specification of where to load the theory from
    ///
    /// # Returns
    /// - `Ok(theory)`: The loaded and compiled theory
    /// - `Err(SpaceError::TheoryParseError)`: If the theory could not be loaded
    fn load(&self, spec: &TheorySpec) -> Result<BoxedTheory, SpaceError>;

    /// Check if this loader can handle the given theory specification.
    ///
    /// Default implementation returns true for all specs.
    fn can_handle(&self, spec: &TheorySpec) -> bool {
        let _ = spec;
        true
    }

    /// Get a list of built-in theory names this loader provides.
    fn builtin_theories(&self) -> Vec<&str> {
        Vec::new()
    }
}

// =============================================================================
// Builtin Theory Loader
// =============================================================================

/// A simple theory loader that provides only built-in theories.
///
/// This is a placeholder implementation until full MeTTaIL integration is available.
/// It provides basic theories like `Nat`, `Int`, `String`, etc.
#[derive(Clone, Debug, Default)]
pub struct BuiltinTheoryLoader;

impl BuiltinTheoryLoader {
    /// Create a new builtin theory loader.
    pub fn new() -> Self {
        BuiltinTheoryLoader
    }

    /// Get a builtin theory by name.
    fn get_builtin(&self, name: &str) -> Option<BoxedTheory> {
        use super::super::types::SimpleTypeTheory;

        match name {
            "Nat" => Some(Box::new(SimpleTypeTheory::new(
                "Nat",
                vec!["Nat".to_string(), "0".to_string()],
            ))),
            "Int" => Some(Box::new(SimpleTypeTheory::new(
                "Int",
                vec!["Int".to_string(), "Nat".to_string()],
            ))),
            "String" => Some(Box::new(SimpleTypeTheory::new(
                "String",
                vec!["String".to_string(), "\"".to_string()],
            ))),
            "Bool" => Some(Box::new(SimpleTypeTheory::new(
                "Bool",
                vec!["Bool".to_string(), "true".to_string(), "false".to_string()],
            ))),
            "Any" => Some(Box::new(super::super::types::NullTheory)),
            _ => None,
        }
    }
}

impl TheoryLoader for BuiltinTheoryLoader {
    fn load(&self, spec: &TheorySpec) -> Result<BoxedTheory, SpaceError> {
        match spec {
            TheorySpec::Builtin(name) => {
                self.get_builtin(name).ok_or_else(|| SpaceError::TheoryNotSupported {
                    theory_name: name.clone(),
                    reason: format!(
                        "Unknown builtin theory '{}'. Available: {:?}",
                        name,
                        self.builtin_theories()
                    ),
                })
            }
            TheorySpec::MeTTaILFile(path) => Err(SpaceError::TheoryNotSupported {
                theory_name: format!("mettail:{}", path),
                reason: "MeTTaIL file loading not yet implemented. Use a MeTTaIL-enabled loader.".to_string(),
            }),
            TheorySpec::InlineMeTTaIL(code) => Err(SpaceError::TheoryNotSupported {
                theory_name: format!("inline:{}", code),
                reason: "Inline MeTTaIL parsing not yet implemented. Use a MeTTaIL-enabled loader.".to_string(),
            }),
            TheorySpec::Uri(uri) => Err(SpaceError::TheoryNotSupported {
                theory_name: format!("uri:{}", uri),
                reason: "URI-based theory loading not yet implemented.".to_string(),
            }),
        }
    }

    fn can_handle(&self, spec: &TheorySpec) -> bool {
        matches!(spec, TheorySpec::Builtin(_))
    }

    fn builtin_theories(&self) -> Vec<&str> {
        vec!["Nat", "Int", "String", "Bool", "Any"]
    }
}

/// Shared theory loader for use across the runtime.
pub type SharedTheoryLoader = Arc<dyn TheoryLoader>;

// =============================================================================
// URN Parsing with Theory
// =============================================================================

/// Parse a URN with an optional theory extension.
///
/// URNs can include theory annotations in square brackets:
/// - `rho:space:HashMapBagSpace[theory=Nat]`
/// - `rho:space:QueueSpace[theory=mettail:types.metta]`
///
/// # Returns
/// - Tuple of (base URN without theory, optional TheorySpec)
///
/// # Examples
/// ```ignore
/// let (base, spec) = parse_urn_with_theory("rho:space:HashMapBagSpace[theory=Nat]");
/// assert_eq!(base, "rho:space:HashMapBagSpace");
/// assert_eq!(spec, Some(TheorySpec::Builtin("Nat".to_string())));
/// ```
pub fn parse_urn_with_theory(urn: &str) -> (String, Option<TheorySpec>) {
    if let Some(bracket_start) = urn.find('[') {
        let base_urn = urn[..bracket_start].to_string();

        // Parse the extension
        if let Some(bracket_end) = urn.find(']') {
            let extension = &urn[bracket_start + 1..bracket_end];

            // Look for theory= parameter
            if let Some(theory_start) = extension.find("theory=") {
                let theory_value = &extension[theory_start + 7..];
                // Handle comma-separated parameters
                let theory_end = theory_value.find(',').unwrap_or(theory_value.len());
                let theory_str = theory_value[..theory_end].trim();

                return (base_urn, Some(TheorySpec::parse(theory_str)));
            }
        }

        (base_urn, None)
    } else {
        (urn.to_string(), None)
    }
}

/// Parse a full URN with theory and return a SpaceConfig with the theory attached.
///
/// This is a convenience function that combines `parse_urn_with_theory` and
/// `config_from_urn` with theory loading.
///
/// # Arguments
/// - `urn`: The full URN including optional theory extension
/// - `loader`: The theory loader to use for loading the theory
///
/// # Returns
/// - `Ok(config)`: SpaceConfig with theory attached if specified
/// - `Err(error)`: If URN parsing or theory loading fails
pub fn config_from_full_urn(
    urn: &str,
    loader: &dyn TheoryLoader,
) -> Result<SpaceConfig, SpaceError> {
    let (base_urn, theory_spec) = parse_urn_with_theory(urn);

    let mut config = config_from_urn(&base_urn).ok_or_else(|| SpaceError::InvalidConfiguration {
        description: format!("Unknown space URN: {}", base_urn),
    })?;

    // Load and attach theory if specified
    if let Some(spec) = theory_spec {
        let theory = loader.load(&spec)?;
        config.theory = Some(theory);
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::types::OuterStorageType;

    #[test]
    fn test_theory_spec_parse_builtin() {
        let spec = TheorySpec::parse("Nat");
        assert_eq!(spec, TheorySpec::Builtin("Nat".to_string()));
    }

    #[test]
    fn test_theory_spec_parse_mettail() {
        let spec = TheorySpec::parse("mettail:types/nat.metta");
        assert_eq!(spec, TheorySpec::MeTTaILFile("types/nat.metta".to_string()));
    }

    #[test]
    fn test_theory_spec_parse_inline() {
        let spec = TheorySpec::parse("inline:(: Nat Type)");
        assert_eq!(spec, TheorySpec::InlineMeTTaIL("(: Nat Type)".to_string()));
    }

    #[test]
    fn test_theory_spec_parse_uri() {
        let spec = TheorySpec::parse("uri:https://example.com/nat.metta");
        assert_eq!(spec, TheorySpec::Uri("https://example.com/nat.metta".to_string()));
    }

    #[test]
    fn test_parse_urn_with_theory_builtin() {
        let (base, spec) = parse_urn_with_theory("rho:space:HashMapBagSpace[theory=Nat]");
        assert_eq!(base, "rho:space:HashMapBagSpace");
        assert_eq!(spec, Some(TheorySpec::Builtin("Nat".to_string())));
    }

    #[test]
    fn test_parse_urn_with_theory_mettail() {
        let (base, spec) = parse_urn_with_theory("rho:space:QueueSpace[theory=mettail:types.metta]");
        assert_eq!(base, "rho:space:QueueSpace");
        assert_eq!(spec, Some(TheorySpec::MeTTaILFile("types.metta".to_string())));
    }

    #[test]
    fn test_parse_urn_without_theory() {
        let (base, spec) = parse_urn_with_theory("rho:space:HashMapBagSpace");
        assert_eq!(base, "rho:space:HashMapBagSpace");
        assert_eq!(spec, None);
    }

    #[test]
    fn test_builtin_theory_loader_nat() {
        let loader = BuiltinTheoryLoader::new();
        let spec = TheorySpec::Builtin("Nat".to_string());

        let theory = loader.load(&spec).expect("Should load Nat theory");
        assert_eq!(theory.name(), "Nat");
    }

    #[test]
    fn test_builtin_theory_loader_any() {
        let loader = BuiltinTheoryLoader::new();
        let spec = TheorySpec::Builtin("Any".to_string());

        let theory = loader.load(&spec).expect("Should load Any theory");
        // Any theory should validate everything
        assert!(theory.validate("anything").is_ok());
    }

    #[test]
    fn test_builtin_theory_loader_unknown() {
        let loader = BuiltinTheoryLoader::new();
        let spec = TheorySpec::Builtin("UnknownTheory".to_string());

        let result = loader.load(&spec);
        assert!(result.is_err());
        if let Err(SpaceError::TheoryNotSupported { theory_name, .. }) = result {
            assert_eq!(theory_name, "UnknownTheory");
        } else {
            panic!("Expected TheoryNotSupported error");
        }
    }

    #[test]
    fn test_builtin_theory_loader_unsupported_file() {
        let loader = BuiltinTheoryLoader::new();
        let spec = TheorySpec::MeTTaILFile("types.metta".to_string());

        let result = loader.load(&spec);
        assert!(result.is_err());
        assert!(matches!(result, Err(SpaceError::TheoryNotSupported { .. })));
    }

    #[test]
    fn test_config_from_full_urn_with_theory() {
        let loader = BuiltinTheoryLoader::new();
        let config = config_from_full_urn(
            "rho:space:HashMapBagSpace[theory=Nat]",
            &loader,
        ).expect("Should parse URN with theory");

        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert!(config.theory.is_some());
        assert_eq!(config.theory.as_ref().expect("Theory exists").name(), "Nat");
    }

    #[test]
    fn test_config_from_full_urn_without_theory() {
        use super::super::super::types::InnerCollectionType;

        let loader = BuiltinTheoryLoader::new();
        let config = config_from_full_urn(
            "rho:space:QueueSpace",
            &loader,
        ).expect("Should parse URN without theory");

        assert_eq!(config.data_collection, InnerCollectionType::Queue);
        assert!(config.theory.is_none());
    }

    #[test]
    fn test_builtin_theory_loader_can_handle() {
        let loader = BuiltinTheoryLoader::new();

        assert!(loader.can_handle(&TheorySpec::Builtin("Nat".to_string())));
        assert!(!loader.can_handle(&TheorySpec::MeTTaILFile("foo.metta".to_string())));
    }

    #[test]
    fn test_builtin_theories_list() {
        let loader = BuiltinTheoryLoader::new();
        let theories = loader.builtin_theories();

        assert!(theories.contains(&"Nat"));
        assert!(theories.contains(&"Int"));
        assert!(theories.contains(&"String"));
        assert!(theories.contains(&"Bool"));
        assert!(theories.contains(&"Any"));
    }
}
