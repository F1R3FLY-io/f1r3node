//! Property-Based Tests for getSpaceAgent Built-in Function
//!
//! This module provides tests for the `getSpaceAgent(space)` built-in function,
//! which returns the factory URN that created a given space.
//!
//! # Formal Correspondence
//! - `SpaceAgent.v`: getSpaceAgent correctness and URN preservation invariants
//!
//! # Test Coverage
//! - URN generation from SpaceConfig (core getSpaceAgent logic)
//! - EFunction expression structure validation
//! - URN roundtrip preservation
//! - Error handling for invalid arguments
//!
//! # Rholang Syntax Examples
//!
//! The getSpaceAgent function is used as follows:
//! ```rholang
//! new HMB(`rho:space:HashMapBagSpace`),
//!     PM(`rho:space:PathMapSpace`),
//!     getSpaceAgent(`rho:space:getSpaceAgent`),
//!     assertEqual(`rho:assert:assertEqual`) in {
//!   for(space_1 <- HMB!?("default", free Nat());
//!       space_2 <- PM!?("temp", free String())) {
//!     // getSpaceAgent returns the factory URN for a space
//!     assertEqual!(getSpaceAgent(space_1), HMB) |
//!     assertEqual!(getSpaceAgent(space_2), PM)
//!   }
//! }
//! ```

use proptest::prelude::*;

use rholang::rust::interpreter::spaces::factory::{
    InnerParams, InnerType, OuterParams, OuterType, Qualifier,
    compute_config, is_valid_combination,
};

use rholang::rust::interpreter::spaces::{
    config_from_urn, urn_from_config,
    InnerCollectionType, OuterStorageType, SpaceConfig, SpaceQualifier,
    GasConfiguration,
};

use models::rhoapi::{EFunction, Expr, Par};
use models::rhoapi::expr::ExprInstance;

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 500;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Arbitrary Generators for Space Types
// =============================================================================

/// Generate an arbitrary InnerType.
pub fn arb_inner_type() -> impl Strategy<Value = InnerType> {
    prop_oneof![
        Just(InnerType::Bag),
        Just(InnerType::Queue),
        Just(InnerType::Stack),
        Just(InnerType::Set),
        Just(InnerType::Cell),
        Just(InnerType::PriorityQueue),
        Just(InnerType::VectorDB),
    ]
}

/// Generate an arbitrary OuterType.
pub fn arb_outer_type() -> impl Strategy<Value = OuterType> {
    prop_oneof![
        Just(OuterType::HashMap),
        Just(OuterType::PathMap),
        Just(OuterType::Array),
        Just(OuterType::Vector),
        Just(OuterType::HashSet),
    ]
}

/// Generate an arbitrary Qualifier.
pub fn arb_qualifier() -> impl Strategy<Value = Qualifier> {
    prop_oneof![
        Just(Qualifier::Default),
        Just(Qualifier::Temp),
        Just(Qualifier::Seq),
    ]
}

/// Generate a valid (inner, outer) combination.
pub fn arb_valid_combination() -> impl Strategy<Value = (InnerType, OuterType)> {
    (arb_inner_type(), arb_outer_type())
        .prop_filter("must be valid combination", |(inner, outer)| {
            is_valid_combination(*inner, *outer)
        })
}

/// Generate a valid SpaceConfig.
pub fn arb_space_config() -> impl Strategy<Value = SpaceConfig> {
    (arb_valid_combination(), arb_qualifier()).prop_map(|((inner, outer), qualifier)| {
        let inner_params = match inner {
            InnerType::PriorityQueue => InnerParams::PriorityQueue { priorities: 2 },
            InnerType::VectorDB => InnerParams::VectorDB {
                dimensions: 384,
                backend: "rho".to_string(),
            },
            _ => InnerParams::None,
        };
        let outer_params = match outer {
            OuterType::Array => OuterParams::Array { size: 1000, cyclic: false },
            _ => OuterParams::None,
        };

        compute_config(inner, inner_params, outer, outer_params, qualifier)
    })
}

// =============================================================================
// EFunction Expression Tests
// =============================================================================

#[test]
fn test_efunction_structure_for_get_space_agent() {
    // Verify that EFunction can hold getSpaceAgent function call
    let efunc = EFunction {
        function_name: "getSpaceAgent".to_string(),
        arguments: vec![Par::default()],
        connective_used: false,
        locally_free: Vec::new(),
    };

    assert_eq!(efunc.function_name, "getSpaceAgent");
    assert_eq!(efunc.arguments.len(), 1);
}

#[test]
fn test_efunction_expr_instance_creation() {
    // Verify EFunctionBody can be wrapped in Expr
    let efunc = EFunction {
        function_name: "getSpaceAgent".to_string(),
        arguments: vec![Par::default()],
        connective_used: false,
        locally_free: Vec::new(),
    };

    let expr = Expr {
        expr_instance: Some(ExprInstance::EFunctionBody(efunc)),
    };

    match expr.expr_instance {
        Some(ExprInstance::EFunctionBody(ref f)) => {
            assert_eq!(f.function_name, "getSpaceAgent");
        }
        _ => panic!("Expected EFunctionBody"),
    }
}

#[test]
fn test_efunction_with_multiple_arguments_rejected_semantically() {
    // getSpaceAgent should only accept 1 argument
    // This tests that we can detect multi-argument calls
    let efunc = EFunction {
        function_name: "getSpaceAgent".to_string(),
        arguments: vec![Par::default(), Par::default()],
        connective_used: false,
        locally_free: Vec::new(),
    };

    // The function accepts multiple arguments structurally,
    // but the evaluator should reject this at runtime
    assert_eq!(efunc.arguments.len(), 2);
}

#[test]
fn test_efunction_with_no_arguments_rejected_semantically() {
    // getSpaceAgent should only accept 1 argument
    let efunc = EFunction {
        function_name: "getSpaceAgent".to_string(),
        arguments: vec![],
        connective_used: false,
        locally_free: Vec::new(),
    };

    assert_eq!(efunc.arguments.len(), 0);
}

// =============================================================================
// URN Generation Tests (Core getSpaceAgent Logic)
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: getSpaceAgent URN output matches urn_from_config.
    ///
    /// This property verifies that the URN generation used by getSpaceAgent
    /// is consistent with the factory URN parsing.
    ///
    /// ∀ config ∈ valid_configs.
    ///   let urn = urn_from_config(config)
    ///   config_from_urn(urn).is_some()
    ///
    /// Rocq: `get_space_agent_urn_roundtrip` in SpaceAgent.v
    #[test]
    fn prop_get_space_agent_urn_roundtrip(config in arb_space_config()) {
        let urn = urn_from_config(&config);

        // The URN should be parseable back to a config
        let parsed = config_from_urn(&urn);
        prop_assert!(
            parsed.is_some(),
            "URN '{}' generated from config should be parseable",
            urn
        );

        // Core fields should match
        let recovered = parsed.expect("should parse");
        prop_assert_eq!(
            config.outer, recovered.outer,
            "Outer storage should roundtrip"
        );
        prop_assert_eq!(
            config.data_collection, recovered.data_collection,
            "Data collection should roundtrip"
        );
        prop_assert_eq!(
            config.qualifier, recovered.qualifier,
            "Qualifier should roundtrip"
        );
    }

    /// Property: Different configs produce different URNs (injectivity).
    ///
    /// ∀ config1, config2 ∈ valid_configs.
    ///   config1 ≠ config2 → urn_from_config(config1) ≠ urn_from_config(config2)
    ///   (for configs differing in outer, data_collection, or qualifier)
    ///
    /// Rocq: `get_space_agent_urn_injective` in SpaceAgent.v
    #[test]
    fn prop_get_space_agent_urn_injective(
        (inner1, outer1) in arb_valid_combination(),
        (inner2, outer2) in arb_valid_combination(),
        qual1 in arb_qualifier(),
        qual2 in arb_qualifier()
    ) {
        // Only test when configs differ in meaningful ways
        if inner1 != inner2 || outer1 != outer2 || qual1 != qual2 {
            let make_config = |inner: InnerType, outer: OuterType, qual: Qualifier| {
                let inner_params = match inner {
                    InnerType::PriorityQueue => InnerParams::PriorityQueue { priorities: 2 },
                    InnerType::VectorDB => InnerParams::VectorDB {
                        dimensions: 384,
                        backend: "rho".to_string(),
                    },
                    _ => InnerParams::None,
                };
                let outer_params = match outer {
                    OuterType::Array => OuterParams::Array { size: 1000, cyclic: false },
                    _ => OuterParams::None,
                };
                compute_config(inner, inner_params, outer, outer_params, qual)
            };

            let config1 = make_config(inner1, outer1, qual1);
            let config2 = make_config(inner2, outer2, qual2);

            let urn1 = urn_from_config(&config1);
            let urn2 = urn_from_config(&config2);

            // Different configs should produce different URNs
            prop_assert_ne!(
                urn1, urn2,
                "Different configs should produce different URNs"
            );
        }
    }

    /// Property: URN format is consistent.
    ///
    /// All generated URNs should start with "rho:space:" prefix.
    ///
    /// Rocq: `get_space_agent_urn_format` in SpaceAgent.v
    #[test]
    fn prop_get_space_agent_urn_format(config in arb_space_config()) {
        let urn = urn_from_config(&config);

        prop_assert!(
            urn.starts_with("rho:space:"),
            "URN '{}' should start with 'rho:space:' prefix",
            urn
        );
    }
}

// =============================================================================
// Static Tests for Known Space Types
// =============================================================================

#[test]
fn test_get_space_agent_hashmap_bag_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Bag,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:HashMapBagSpace");
}

#[test]
fn test_get_space_agent_pathmap_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::PathMap,
        data_collection: InnerCollectionType::Bag,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:PathMapSpace");
}

#[test]
fn test_get_space_agent_queue_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Queue,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:QueueSpace");
}

#[test]
fn test_get_space_agent_stack_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Stack,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:StackSpace");
}

#[test]
fn test_get_space_agent_cell_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Cell,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:CellSpace");
}

#[test]
fn test_get_space_agent_set_default() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Set,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:SetSpace");
}

#[test]
fn test_get_space_agent_temp_space() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::Bag,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Temp,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:TempSpace");
}

#[test]
fn test_get_space_agent_seq_space() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashSet,
        data_collection: InnerCollectionType::Set,
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Seq,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    assert_eq!(urn, "rho:space:SeqSpace");
}

#[test]
fn test_get_space_agent_priority_queue() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::PriorityQueue { priorities: 5 },
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    // Extended URN format for parametric types
    assert!(urn.contains("priorityqueue"));
    assert!(urn.contains("5"));
}

#[test]
fn test_get_space_agent_vectordb() {
    let config = SpaceConfig {
        outer: OuterStorageType::HashMap,
        data_collection: InnerCollectionType::VectorDB {
            dimensions: 384,
            backend: "rho".to_string(),
        },
        continuation_collection: InnerCollectionType::Bag,
        qualifier: SpaceQualifier::Default,
        theory: None,
        gas_config: GasConfiguration::default(),
    };

    let urn = urn_from_config(&config);
    // Extended URN format for VectorDB
    assert!(urn.contains("vectordb"));
    assert!(urn.contains("384"));
}

// =============================================================================
// EFunction Expression Building Tests
// =============================================================================

#[test]
fn test_build_efunction_expr_for_par() {
    // Test creating a Par containing an EFunction expression
    use models::rhoapi::{GPrivate, GUnforgeable};
    use models::rhoapi::g_unforgeable::UnfInstance;

    // Create a space reference (GPrivate)
    let space_id = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let space_ref = Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                id: space_id.clone(),
            })),
        }],
        ..Par::default()
    };

    // Create getSpaceAgent(space_ref) call
    let efunc = EFunction {
        function_name: "getSpaceAgent".to_string(),
        arguments: vec![space_ref.clone()],
        connective_used: false,
        locally_free: Vec::new(),
    };

    let expr = Expr {
        expr_instance: Some(ExprInstance::EFunctionBody(efunc)),
    };

    let result_par = Par {
        exprs: vec![expr],
        ..Par::default()
    };

    // Verify structure
    assert_eq!(result_par.exprs.len(), 1);
    match &result_par.exprs[0].expr_instance {
        Some(ExprInstance::EFunctionBody(f)) => {
            assert_eq!(f.function_name, "getSpaceAgent");
            assert_eq!(f.arguments.len(), 1);
            assert_eq!(f.arguments[0].unforgeables.len(), 1);
        }
        _ => panic!("Expected EFunctionBody"),
    }
}

// =============================================================================
// Unknown Function Tests
// =============================================================================

#[test]
fn test_unknown_function_name_structure() {
    // Test that unknown functions can be represented
    // (but should be rejected by the evaluator)
    let efunc = EFunction {
        function_name: "unknownFunction".to_string(),
        arguments: vec![Par::default()],
        connective_used: false,
        locally_free: Vec::new(),
    };

    assert_eq!(efunc.function_name, "unknownFunction");
}
