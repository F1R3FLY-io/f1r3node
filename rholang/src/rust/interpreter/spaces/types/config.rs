//! Space Configuration
//!
//! This module defines SpaceConfig and GasConfiguration for configuring
//! space behavior, storage types, and resource metering.

use std::fmt;
use std::str::FromStr;

use super::allocation::AllocationMode;
use super::collections::{InnerCollectionType, OuterStorageType};
use super::qualifier::SpaceQualifier;
use super::theory::BoxedTheory;

// ==========================================================================
// Gas/Phlogiston Configuration
// ==========================================================================

/// Configuration for phlogiston (gas) accounting in a space.
///
/// This determines whether and how resource consumption is metered for
/// operations within the space.
///
/// # Formal Correspondence
/// - `Phlogiston.v`: Gas accounting invariants
/// - `Safety/Properties.v`: Resource exhaustion safety
#[derive(Clone, Debug)]
pub struct GasConfiguration {
    /// Whether gas accounting is enabled.
    pub enabled: bool,

    /// Initial gas limit for new operations.
    pub initial_limit: u64,

    /// Cost multiplier for chain economics.
    pub cost_multiplier: f64,
}

impl Default for GasConfiguration {
    fn default() -> Self {
        GasConfiguration {
            enabled: true,
            initial_limit: 10_000_000,
            cost_multiplier: 1.0,
        }
    }
}

impl GasConfiguration {
    /// Create a configuration with gas accounting disabled.
    pub fn disabled() -> Self {
        GasConfiguration {
            enabled: false,
            initial_limit: u64::MAX,
            cost_multiplier: 1.0,
        }
    }

    /// Create a configuration with unlimited gas (for testing).
    pub fn unlimited() -> Self {
        GasConfiguration {
            enabled: true,
            initial_limit: u64::MAX,
            cost_multiplier: 1.0,
        }
    }

    /// Create a configuration with a specific limit.
    pub fn with_limit(limit: u64) -> Self {
        GasConfiguration {
            enabled: true,
            initial_limit: limit,
            cost_multiplier: 1.0,
        }
    }

    /// Set the cost multiplier.
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.cost_multiplier = multiplier;
        self
    }
}

// ==========================================================================
// Space Configuration
// ==========================================================================

/// Full space configuration combining outer and inner types.
///
/// This struct captures all the configuration needed to create a space with
/// specific storage and behavior characteristics.
///
/// # Theory Integration
/// The optional `theory` field allows spaces to validate data against a type
/// theory before accepting it. This enables:
/// - Typed tuple spaces where only well-typed data can be stored
/// - Contract validation for smart contract execution
/// - Schema enforcement for structured data
///
/// When a theory is present, `produce` operations validate data against the
/// theory before storing, rejecting invalid data with `SpaceError::TheoryValidationError`.
///
/// # Gas/Phlogiston Accounting
/// The `gas_config` field enables resource consumption metering.
/// Operations consume phlogiston based on their cost, and
/// `OutOfPhlogiston` errors are raised when limits are exceeded.
pub struct SpaceConfig {
    /// Outer storage structure (channel indexing)
    pub outer: OuterStorageType,

    /// Inner collection type for data at channels
    pub data_collection: InnerCollectionType,

    /// Inner collection type for continuations at channels
    pub continuation_collection: InnerCollectionType,

    /// Qualifier for persistence/concurrency behavior
    pub qualifier: SpaceQualifier,

    /// Optional theory for data validation (MeTTaIL integration).
    ///
    /// When present, data is validated against this theory before being
    /// accepted by the space. This enables typed tuple spaces.
    pub theory: Option<BoxedTheory>,

    /// Gas/phlogiston configuration.
    ///
    /// Operations consume phlogiston and are rejected if the limit is exceeded.
    pub gas_config: GasConfiguration,
}

impl fmt::Debug for SpaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpaceConfig")
            .field("outer", &self.outer)
            .field("data_collection", &self.data_collection)
            .field("continuation_collection", &self.continuation_collection)
            .field("qualifier", &self.qualifier)
            .field("theory", &self.theory.as_ref().map(|t| t.name()))
            .field("gas_config", &self.gas_config)
            .finish()
    }
}

impl Clone for SpaceConfig {
    fn clone(&self) -> Self {
        SpaceConfig {
            outer: self.outer.clone(),
            data_collection: self.data_collection.clone(),
            continuation_collection: self.continuation_collection.clone(),
            qualifier: self.qualifier,
            theory: self.theory.as_ref().map(|t| t.clone_box()),
            gas_config: self.gas_config.clone(),
        }
    }
}

impl Default for SpaceConfig {
    fn default() -> Self {
        SpaceConfig {
            outer: OuterStorageType::PathMap,
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }
}

impl SpaceConfig {
    // ======================================================================
    // Pre-defined space configurations matching spec URNs
    // ======================================================================

    /// `rho:space:HashMapBagSpace` - HashMap outer + Bag inner (original default)
    pub fn hashmap_bag() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:PathMapSpace` - PathMap outer + Bag inner (recommended default)
    pub fn pathmap() -> Self {
        SpaceConfig {
            outer: OuterStorageType::PathMap,
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:QueueSpace` - HashMap outer + Queue inner (FIFO)
    pub fn queue() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Queue,
            continuation_collection: InnerCollectionType::Queue,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:StackSpace` - HashMap outer + Stack inner (LIFO)
    pub fn stack() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Stack,
            continuation_collection: InnerCollectionType::Stack,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:SetSpace` - HashMap outer + Set inner (idempotent)
    pub fn set() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Set,
            continuation_collection: InnerCollectionType::Set,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:CellSpace` - HashMap outer + Cell inner (exactly-once)
    pub fn cell() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Cell,
            continuation_collection: InnerCollectionType::Cell,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:ArraySpace` - Array outer + Bag inner (fixed size)
    pub fn array(max_size: usize, cyclic: bool) -> Self {
        SpaceConfig {
            outer: OuterStorageType::Array { max_size, cyclic },
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:VectorSpace` - Vector outer + Bag inner (unbounded)
    pub fn vector() -> Self {
        SpaceConfig {
            outer: OuterStorageType::Vector,
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:SeqSpace` - HashSet outer + Set inner (sequential)
    /// For sequential processes with restricted channel mobility.
    pub fn seq() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashSet,
            data_collection: InnerCollectionType::Set,
            continuation_collection: InnerCollectionType::Set,
            qualifier: SpaceQualifier::Seq,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:TempSpace` - HashMap outer + Bag inner (non-persistent)
    pub fn temp() -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::Bag,
            continuation_collection: InnerCollectionType::Bag,
            qualifier: SpaceQualifier::Temp,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:PriorityQueueSpace` - HashMap outer + PriorityQueue inner
    pub fn priority_queue(priorities: usize) -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::PriorityQueue { priorities },
            continuation_collection: InnerCollectionType::PriorityQueue { priorities },
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    /// `rho:space:VectorDBSpace` - HashMap outer + VectorDB inner (similarity matching)
    ///
    /// Uses the default "rho" backend (in-memory SIMD-optimized).
    pub fn vector_db(dimensions: usize) -> Self {
        Self::vector_db_with_backend(dimensions, "rho".to_string())
    }

    /// `rho:space:VectorDBSpace` with explicit backend specification.
    ///
    /// # Arguments
    /// * `dimensions` - Dimensionality of embedding vectors
    /// * `backend` - Backend name (e.g., "rho", "pinecone"). Looked up in BackendRegistry.
    ///
    /// # Example
    /// ```ignore
    /// // In Rholang:
    /// // VectorDBFactory!({"dimensions": 384, "backend": "pinecone", ...}, *space)
    ///
    /// let config = SpaceConfig::vector_db_with_backend(384, "pinecone".to_string());
    /// ```
    pub fn vector_db_with_backend(dimensions: usize, backend: String) -> Self {
        SpaceConfig {
            outer: OuterStorageType::HashMap,
            data_collection: InnerCollectionType::VectorDB { dimensions, backend },
            continuation_collection: InnerCollectionType::Bag, // Continuations use Bag
            qualifier: SpaceQualifier::Default,
            theory: None,
            gas_config: GasConfiguration::default(),
        }
    }

    // ======================================================================
    // Builder methods for custom configurations
    // ======================================================================

    /// Set the outer storage type.
    pub fn with_outer(mut self, outer: OuterStorageType) -> Self {
        self.outer = outer;
        self
    }

    /// Set the data collection type.
    pub fn with_data_collection(mut self, collection: InnerCollectionType) -> Self {
        self.data_collection = collection;
        self
    }

    /// Set the continuation collection type.
    pub fn with_continuation_collection(mut self, collection: InnerCollectionType) -> Self {
        self.continuation_collection = collection;
        self
    }

    /// Set the space qualifier.
    pub fn with_qualifier(mut self, qualifier: SpaceQualifier) -> Self {
        self.qualifier = qualifier;
        self
    }

    /// Set the theory for data validation.
    ///
    /// When a theory is set, all data entering the space via `produce` will
    /// be validated against the theory. Invalid data will be rejected with
    /// a `SpaceError::TheoryValidationError`.
    ///
    /// # Example
    /// ```ignore
    /// let theory = SimpleTypeTheory::new("NatTheory", vec!["Nat".to_string()]);
    /// let config = SpaceConfig::default().with_theory(Box::new(theory));
    /// ```
    pub fn with_theory(mut self, theory: BoxedTheory) -> Self {
        self.theory = Some(theory);
        self
    }

    /// Clear the theory (remove validation).
    pub fn without_theory(mut self) -> Self {
        self.theory = None;
        self
    }

    /// Set the gas/phlogiston configuration.
    ///
    /// When gas configuration is enabled, operations consume phlogiston
    /// and are rejected if the limit is exceeded.
    ///
    /// # Example
    /// ```ignore
    /// let config = SpaceConfig::default()
    ///     .with_gas(GasConfiguration::with_limit(1_000_000));
    /// ```
    pub fn with_gas(mut self, gas_config: GasConfiguration) -> Self {
        self.gas_config = gas_config;
        self
    }

    /// Set gas configuration with a specific limit.
    pub fn with_gas_limit(mut self, limit: u64) -> Self {
        self.gas_config = GasConfiguration::with_limit(limit);
        self
    }

    /// Enable unlimited gas (for testing).
    pub fn with_unlimited_gas(mut self) -> Self {
        self.gas_config = GasConfiguration::unlimited();
        self
    }

    /// Disable gas accounting.
    pub fn with_disabled_gas(mut self) -> Self {
        self.gas_config = GasConfiguration::disabled();
        self
    }

    /// Check if this space configuration supports persistence.
    pub fn is_persistent(&self) -> bool {
        matches!(self.qualifier, SpaceQualifier::Default)
    }

    /// Check if this space configuration supports concurrent access.
    pub fn is_concurrent(&self) -> bool {
        !matches!(self.qualifier, SpaceQualifier::Seq)
    }

    /// Check if channels in this space can be sent to other processes.
    pub fn is_mobile(&self) -> bool {
        !matches!(self.qualifier, SpaceQualifier::Seq)
    }

    /// Check if this space has a theory for data validation.
    pub fn has_theory(&self) -> bool {
        self.theory.is_some()
    }

    /// Get the theory name if one is set.
    pub fn theory_name(&self) -> Option<&str> {
        self.theory.as_ref().map(|t| t.name())
    }

    /// Validate data against the theory (if present).
    ///
    /// Returns `Ok(())` if no theory is set or if the data validates.
    /// Returns `Err(description)` if validation fails.
    pub fn validate_data(&self, term: &str) -> Result<(), String> {
        match &self.theory {
            Some(theory) => theory.validate(term),
            None => Ok(()),
        }
    }

    /// Check if gas accounting is enabled.
    pub fn has_gas(&self) -> bool {
        self.gas_config.enabled
    }

    /// Get the gas limit.
    pub fn gas_limit(&self) -> u64 {
        self.gas_config.initial_limit
    }

    /// Get the gas configuration.
    pub fn gas(&self) -> &GasConfiguration {
        &self.gas_config
    }

    /// Get the allocation mode for `new` bindings within this space.
    ///
    /// The allocation mode is derived from the outer storage type:
    /// - HashMap, PathMap, HashSet: Random allocation using Blake2b512Random
    /// - Array: Sequential indices up to max_size, wrapped in Unforgeable
    /// - Vector: Growing indices, wrapped in Unforgeable
    ///
    /// This determines how `new` bindings allocate channel names when
    /// executing inside a `use space { ... }` block.
    pub fn allocation_mode(&self) -> AllocationMode {
        match &self.outer {
            OuterStorageType::Array { max_size, cyclic } => {
                AllocationMode::ArrayIndex {
                    max_size: *max_size,
                    cyclic: *cyclic,
                }
            }
            OuterStorageType::Vector => AllocationMode::VectorIndex,
            OuterStorageType::HashMap
            | OuterStorageType::PathMap
            | OuterStorageType::HashSet => AllocationMode::Random,
        }
    }

    // ======================================================================
    // Configuration Validation
    // ======================================================================

    /// Validate that this configuration is internally consistent.
    ///
    /// This checks for invalid combinations of outer storage, inner collections,
    /// and qualifiers. Call this after building a configuration to ensure it's valid.
    ///
    /// # Invalid Combinations
    ///
    /// - `Seq` qualifier with `PathMap` outer storage: Seq requires restricted
    ///   channel mobility which PathMap doesn't enforce well.
    /// - `VectorDB` collection with `PathMap` outer: VectorDB requires HashMap-style
    ///   channel indexing for embedding lookups.
    /// - `Cell` collection with `PathMap` outer: Cell's exactly-once semantics
    ///   don't align well with PathMap's prefix aggregation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // This will fail validation
    /// let result = SpaceConfig::seq()
    ///     .with_outer(OuterStorageType::PathMap)
    ///     .validate();
    /// assert!(result.is_err());
    ///
    /// // This is valid
    /// let config = SpaceConfig::hashmap_bag().validate()?;
    /// ```
    pub fn validate(&self) -> Result<Self, super::super::errors::SpaceError>
    where
        Self: Clone,
    {
        // Check: Seq qualifier cannot use PathMap
        if self.qualifier == SpaceQualifier::Seq
            && matches!(self.outer, OuterStorageType::PathMap)
        {
            return Err(super::super::errors::SpaceError::InvalidConfiguration {
                description: "Seq qualifier cannot use PathMap storage: Seq requires \
                              restricted channel mobility which PathMap doesn't enforce"
                    .to_string(),
            });
        }

        // Check: VectorDB collection requires HashMap-compatible outer storage
        if matches!(self.data_collection, InnerCollectionType::VectorDB { .. }) {
            match self.outer {
                OuterStorageType::HashMap | OuterStorageType::HashSet | OuterStorageType::Vector => {
                    // OK: These support direct channel lookup needed for embedding queries
                }
                OuterStorageType::PathMap | OuterStorageType::Array { .. } => {
                    return Err(super::super::errors::SpaceError::InvalidConfiguration {
                        description: format!(
                            "VectorDB collection cannot use {} outer storage: VectorDB \
                             requires HashMap-style channel indexing for embedding lookups",
                            self.outer
                        ),
                    });
                }
            }
        }

        // Check: Cell collection (exactly-once) doesn't work well with PathMap aggregation
        if matches!(self.data_collection, InnerCollectionType::Cell)
            && matches!(self.outer, OuterStorageType::PathMap)
        {
            return Err(super::super::errors::SpaceError::InvalidConfiguration {
                description: "Cell collection cannot use PathMap outer storage: Cell's \
                              exactly-once semantics conflict with PathMap's prefix aggregation"
                    .to_string(),
            });
        }

        // Check: PriorityQueue needs at least 1 priority level
        if let InnerCollectionType::PriorityQueue { priorities } = self.data_collection {
            if priorities == 0 {
                return Err(super::super::errors::SpaceError::InvalidConfiguration {
                    description: "PriorityQueue must have at least 1 priority level".to_string(),
                });
            }
        }

        // Check: VectorDB needs at least 1 dimension
        if let InnerCollectionType::VectorDB { dimensions, .. } = self.data_collection {
            if dimensions == 0 {
                return Err(super::super::errors::SpaceError::InvalidConfiguration {
                    description: "VectorDB must have at least 1 dimension".to_string(),
                });
            }
        }

        // Check: Array with max_size 0 is useless
        if let OuterStorageType::Array { max_size, .. } = self.outer {
            if max_size == 0 {
                return Err(super::super::errors::SpaceError::InvalidConfiguration {
                    description: "Array outer storage must have max_size > 0".to_string(),
                });
            }
        }

        Ok(self.clone())
    }

    /// Validate this configuration, consuming self on success.
    ///
    /// This is a consuming version of `validate()` for use in builder chains
    /// where you want to propagate ownership.
    pub fn validated(self) -> Result<Self, super::super::errors::SpaceError> {
        self.validate()
    }
}

impl fmt::Display for SpaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref theory) = self.theory {
            write!(
                f,
                "SpaceConfig({}, data={}, cont={}, {}, theory={})",
                self.outer, self.data_collection, self.continuation_collection,
                self.qualifier, theory.name()
            )
        } else {
            write!(
                f,
                "SpaceConfig({}, data={}, cont={}, {})",
                self.outer, self.data_collection, self.continuation_collection, self.qualifier
            )
        }
    }
}

impl FromStr for SpaceConfig {
    type Err = super::super::errors::SpaceError;

    /// Parse a SpaceConfig from a URN string.
    ///
    /// This enables idiomatic Rust parsing:
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Using parse()
    /// let config: SpaceConfig = "rho:space:HashMapBagSpace".parse()?;
    ///
    /// // Using parse() with extended format
    /// let config: SpaceConfig = "rho:space:queue:pathmap:default".parse()?;
    ///
    /// // Using FromStr::from_str()
    /// let config = SpaceConfig::from_str("rho:space:QueueSpace")?;
    /// ```
    ///
    /// # Supported URN Formats
    ///
    /// ## Short Format (legacy)
    /// - `rho:space:HashMapBagSpace` - HashMap + Bag
    /// - `rho:space:PathMapSpace` - PathMap + Bag
    /// - `rho:space:QueueSpace` - HashMap + Queue
    /// - `rho:space:StackSpace` - HashMap + Stack
    /// - `rho:space:SetSpace` - HashMap + Set
    /// - `rho:space:CellSpace` - HashMap + Cell
    /// - `rho:space:VectorSpace` - Vector + Bag
    /// - `rho:space:SeqSpace` - HashSet + Set (sequential)
    /// - `rho:space:TempSpace` - HashMap + Bag (non-persistent)
    /// - `rho:space:ArraySpace(n,cyclic)` - Array + Bag
    /// - `rho:space:PriorityQueueSpace(n)` - HashMap + PriorityQueue
    /// - `rho:space:VectorDBSpace(dims)` - HashMap + VectorDB
    ///
    /// ## Extended Format: `rho:space:{inner}:{outer}:{qualifier}`
    /// - Inner: bag, queue, stack, set, cell, priorityqueue, vectordb
    /// - Outer: hashmap, pathmap, array, vector, hashset
    /// - Qualifier: default, temp, seq
    ///
    /// # Errors
    ///
    /// Returns `SpaceError::InvalidConfiguration` if the URN is invalid or
    /// represents an unsupported combination.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        super::super::factory::config_from_urn(s).ok_or_else(|| {
            super::super::errors::SpaceError::InvalidConfiguration {
                description: format!("Unknown or invalid space URN: {}", s),
            }
        })
    }
}

/// Convenience trait implementation for converting string references.
impl TryFrom<&str> for SpaceConfig {
    type Error = super::super::errors::SpaceError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.parse()
    }
}

/// Convenience trait implementation for converting owned strings.
impl TryFrom<String> for SpaceConfig {
    type Error = super::super::errors::SpaceError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::theory::{NullTheory, SimpleTypeTheory};

    #[test]
    fn test_space_config_defaults() {
        let config = SpaceConfig::default();
        assert_eq!(config.outer, OuterStorageType::PathMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);
        assert!(config.is_persistent());
        assert!(config.is_concurrent());
        assert!(config.is_mobile());
    }

    #[test]
    fn test_space_config_seq() {
        let config = SpaceConfig::seq();
        assert_eq!(config.outer, OuterStorageType::HashSet);
        assert_eq!(config.qualifier, SpaceQualifier::Seq);
        assert!(!config.is_persistent());
        assert!(!config.is_concurrent());
        assert!(!config.is_mobile());
    }

    #[test]
    fn test_space_config_builder() {
        let config = SpaceConfig::default()
            .with_outer(OuterStorageType::HashMap)
            .with_data_collection(InnerCollectionType::Queue)
            .with_qualifier(SpaceQualifier::Temp);

        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Queue);
        assert_eq!(config.qualifier, SpaceQualifier::Temp);
    }

    #[test]
    fn test_space_config_with_theory() {
        let theory = SimpleTypeTheory::new("TestTheory", vec!["Int".to_string()]);
        let config = SpaceConfig::default().with_theory(Box::new(theory));

        assert!(config.has_theory());
        assert_eq!(config.theory_name(), Some("TestTheory"));

        assert!(config.validate_data("Int 123").is_ok());
        assert!(config.validate_data("String hello").is_err());
    }

    #[test]
    fn test_space_config_without_theory() {
        let config = SpaceConfig::default();

        assert!(!config.has_theory());
        assert_eq!(config.theory_name(), None);

        assert!(config.validate_data("anything").is_ok());
    }

    #[test]
    fn test_theory_builder_pattern() {
        let theory = NullTheory;
        let config = SpaceConfig::hashmap_bag()
            .with_qualifier(SpaceQualifier::Temp)
            .with_theory(Box::new(theory));

        assert!(config.has_theory());
        assert_eq!(config.qualifier, SpaceQualifier::Temp);

        let config2 = SpaceConfig::pathmap()
            .with_theory(Box::new(NullTheory))
            .without_theory();

        assert!(!config2.has_theory());
    }

    #[test]
    fn test_validate_seq_with_pathmap_fails() {
        let config = SpaceConfig::seq().with_outer(OuterStorageType::PathMap);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_vectordb_with_pathmap_fails() {
        let config = SpaceConfig::vector_db(384).with_outer(OuterStorageType::PathMap);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_cell_with_pathmap_fails() {
        let config = SpaceConfig::cell().with_outer(OuterStorageType::PathMap);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_priority_queue_zero_priorities_fails() {
        let config = SpaceConfig::default()
            .with_data_collection(InnerCollectionType::PriorityQueue { priorities: 0 });
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_vectordb_zero_dimensions_fails() {
        let config = SpaceConfig::vector_db_with_backend(0, "rho".to_string());
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_array_zero_size_fails() {
        let config = SpaceConfig::default().with_outer(OuterStorageType::Array {
            max_size: 0,
            cyclic: false,
        });
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_valid_configs_pass() {
        assert!(SpaceConfig::hashmap_bag().validate().is_ok());
        assert!(SpaceConfig::pathmap().validate().is_ok());
        assert!(SpaceConfig::queue().validate().is_ok());
        assert!(SpaceConfig::stack().validate().is_ok());
        assert!(SpaceConfig::set().validate().is_ok());
        assert!(SpaceConfig::cell().validate().is_ok());
        assert!(SpaceConfig::seq().validate().is_ok());
        assert!(SpaceConfig::temp().validate().is_ok());
        assert!(SpaceConfig::vector().validate().is_ok());
        assert!(SpaceConfig::array(10, false).validate().is_ok());
        assert!(SpaceConfig::priority_queue(3).validate().is_ok());
        assert!(SpaceConfig::vector_db(384).validate().is_ok());
    }

    #[test]
    fn test_validated_consumes_and_returns_config() {
        let config = SpaceConfig::hashmap_bag();
        let validated = config.validated().expect("validation should pass");
        assert_eq!(validated.outer, OuterStorageType::HashMap);
        assert_eq!(validated.data_collection, InnerCollectionType::Bag);
    }

    #[test]
    fn test_gas_configuration() {
        let config = SpaceConfig::default().with_gas_limit(1_000_000);
        assert!(config.has_gas());
        assert_eq!(config.gas_limit(), 1_000_000);

        let config2 = SpaceConfig::default().with_unlimited_gas();
        assert!(config2.has_gas());
        assert_eq!(config2.gas_limit(), u64::MAX);

        let config3 = SpaceConfig::default().with_disabled_gas();
        assert!(!config3.has_gas());
    }

    // =========================================================================
    // FromStr / TryFrom Tests
    // =========================================================================

    #[test]
    fn test_from_str_short_format() {
        // Test parsing short-format URNs
        let config: SpaceConfig = "rho:space:HashMapBagSpace".parse().expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);

        let config: SpaceConfig = "rho:space:QueueSpace".parse().expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Queue);

        let config: SpaceConfig = "rho:space:StackSpace".parse().expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Stack);

        let config: SpaceConfig = "rho:space:SetSpace".parse().expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Set);

        let config: SpaceConfig = "rho:space:CellSpace".parse().expect("Should parse");
        assert_eq!(config.data_collection, InnerCollectionType::Cell);
    }

    #[test]
    fn test_from_str_extended_format() {
        // Test parsing extended-format URNs
        let config: SpaceConfig = "rho:space:bag:hashmap:default".parse().expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::HashMap);
        assert_eq!(config.data_collection, InnerCollectionType::Bag);

        let config: SpaceConfig = "rho:space:queue:pathmap:temp".parse().expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::PathMap);
        assert_eq!(config.data_collection, InnerCollectionType::Queue);
        assert_eq!(config.qualifier, SpaceQualifier::Temp);
    }

    #[test]
    fn test_from_str_parametric() {
        // Test parsing parametric URNs
        let config: SpaceConfig = "rho:space:ArraySpace(500,true)".parse().expect("Should parse");
        match config.outer {
            OuterStorageType::Array { max_size, cyclic } => {
                assert_eq!(max_size, 500);
                assert!(cyclic);
            }
            _ => panic!("Expected Array outer type"),
        }

        let config: SpaceConfig = "rho:space:PriorityQueueSpace(4)".parse().expect("Should parse");
        match config.data_collection {
            InnerCollectionType::PriorityQueue { priorities } => {
                assert_eq!(priorities, 4);
            }
            _ => panic!("Expected PriorityQueue collection type"),
        }
    }

    #[test]
    fn test_from_str_invalid_urn() {
        // Test that invalid URNs return errors
        let result: Result<SpaceConfig, _> = "invalid".parse();
        assert!(result.is_err());

        let result: Result<SpaceConfig, _> = "rho:space:UnknownSpace".parse();
        assert!(result.is_err());

        let result: Result<SpaceConfig, _> = "".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_str() {
        // Test TryFrom<&str> implementation
        let config = SpaceConfig::try_from("rho:space:HashMapBagSpace").expect("Should convert");
        assert_eq!(config.outer, OuterStorageType::HashMap);

        let result = SpaceConfig::try_from("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_string() {
        // Test TryFrom<String> implementation
        let urn = String::from("rho:space:QueueSpace");
        let config = SpaceConfig::try_from(urn).expect("Should convert");
        assert_eq!(config.data_collection, InnerCollectionType::Queue);

        let result = SpaceConfig::try_from(String::from("invalid"));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_str_using_from_str_method() {
        // Test using FromStr::from_str() directly
        let config = SpaceConfig::from_str("rho:space:PathMapSpace").expect("Should parse");
        assert_eq!(config.outer, OuterStorageType::PathMap);
    }
}
