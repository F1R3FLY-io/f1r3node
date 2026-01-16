//! Property-Based Tests for Space Factory Module
//!
//! This module provides comprehensive property-based testing for URN parsing,
//! space configuration, and byte name mapping in the factory module.
//!
//! # Formal Correspondence
//! - `SpaceFactory.v`: URN parsing correctness and config generation invariants
//!
//! # Test Coverage
//! - URN parsing roundtrips (InnerType, OuterType, Qualifier)
//! - Valid/invalid combination detection
//! - Byte name uniqueness and determinism
//! - Configuration completeness
//! - Theory specification parsing
//!
//! # Rholang Syntax Examples
//!
//! The factory module parses URNs like:
//! ```rholang
//! new QueueSpace(`rho:space:queue:hashmap:default`), q in {
//!   QueueSpace!({}, *q) |
//!   use q { ... }
//! }
//!
//! // With theory annotation
//! new TypedSpace(`rho:space:HashMapBagSpace[theory=Nat]`), s in {
//!   TypedSpace!({}, *s) |
//!   use s {
//!     // Only accepts natural numbers
//!     channel!(42)  // OK
//!     // channel!(-1)  // Would fail validation
//!   }
//! }
//! ```

use proptest::prelude::*;
use std::collections::HashSet;

// Import from the factory module - use the internal factory module path
// since not all functions are re-exported at the spaces module level
use rholang::rust::interpreter::spaces::factory::{
    InnerType, OuterType, Qualifier,
    InnerParams, OuterParams,
    is_valid_combination, all_valid_urns, valid_urn_count,
    urn_to_byte_name, byte_name_to_urn,
    parse_inner_with_params, parse_outer_with_params,
    compute_config,
};

// These are re-exported at the spaces module level
use rholang::rust::interpreter::spaces::{
    config_from_urn, urn_from_config, parse_urn_with_theory,
    TheorySpec, TheoryLoader, BuiltinTheoryLoader, config_from_full_urn,
    OuterStorageType,
};

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 500;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Arbitrary Generators for URN Types
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

/// Generate an invalid (inner, outer) combination.
pub fn arb_invalid_combination() -> impl Strategy<Value = (InnerType, OuterType)> {
    // VectorDB with incompatible outer types
    prop_oneof![
        Just((InnerType::VectorDB, OuterType::PathMap)),
        Just((InnerType::VectorDB, OuterType::Array)),
        Just((InnerType::VectorDB, OuterType::HashSet)),
    ]
}

/// Generate a valid extended URN string.
pub fn arb_valid_urn() -> impl Strategy<Value = String> {
    (arb_valid_combination(), arb_qualifier()).prop_map(|((inner, outer), qualifier)| {
        format!(
            "rho:space:{}:{}:{}",
            inner.as_str(),
            outer.as_str(),
            qualifier.as_str()
        )
    })
}

/// Generate inner params for a given InnerType.
#[allow(dead_code)]
pub fn arb_inner_params(inner: InnerType) -> impl Strategy<Value = InnerParams> {
    match inner {
        InnerType::PriorityQueue => {
            (1usize..=10).prop_map(|p| InnerParams::PriorityQueue { priorities: p }).boxed()
        }
        InnerType::VectorDB => {
            (2usize..=512).prop_map(|d| InnerParams::VectorDB {
                dimensions: d,
                backend: "rho".to_string(),
            }).boxed()
        }
        _ => Just(InnerParams::None).boxed(),
    }
}

/// Generate outer params for a given OuterType.
#[allow(dead_code)]
pub fn arb_outer_params(outer: OuterType) -> impl Strategy<Value = OuterParams> {
    match outer {
        OuterType::Array => {
            (10usize..=10000, any::<bool>())
                .prop_map(|(size, cyclic)| OuterParams::Array { size, cyclic })
                .boxed()
        }
        _ => Just(OuterParams::None).boxed(),
    }
}

/// Generate a TheorySpec.
#[allow(dead_code)]
pub fn arb_theory_spec() -> impl Strategy<Value = TheorySpec> {
    prop_oneof![
        "[a-zA-Z]{3,10}".prop_map(TheorySpec::Builtin),
        "[a-z/]+\\.metta".prop_map(TheorySpec::MeTTaILFile),
        "\\(: [A-Z][a-z]+ Type\\)".prop_map(TheorySpec::InlineMeTTaIL),
        "https://[a-z]+\\.[a-z]+/[a-z]+".prop_map(TheorySpec::Uri),
    ]
}

/// Generate a builtin theory name.
pub fn arb_builtin_theory_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("Nat".to_string()),
        Just("Int".to_string()),
        Just("String".to_string()),
        Just("Bool".to_string()),
        Just("Any".to_string()),
    ]
}

// =============================================================================
// URN Parsing Roundtrip Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: InnerType::from_str() roundtrips with Display.
    ///
    /// ∀ inner ∈ InnerType. InnerType::from_str(inner.as_str()) == Some(inner)
    ///
    /// Rocq: `inner_type_parse_roundtrip` in SpaceFactory.v
    #[test]
    fn prop_inner_type_parse_roundtrip(inner in arb_inner_type()) {
        let s = inner.as_str();
        let parsed = InnerType::from_str(s);
        prop_assert_eq!(parsed, Some(inner));
    }

    /// Property: OuterType::from_str() roundtrips with Display.
    ///
    /// ∀ outer ∈ OuterType. OuterType::from_str(outer.as_str()) == Some(outer)
    ///
    /// Rocq: `outer_type_parse_roundtrip` in SpaceFactory.v
    #[test]
    fn prop_outer_type_parse_roundtrip(outer in arb_outer_type()) {
        let s = outer.as_str();
        let parsed = OuterType::from_str(s);
        prop_assert_eq!(parsed, Some(outer));
    }

    /// Property: Qualifier::from_str() roundtrips with Display.
    ///
    /// ∀ q ∈ Qualifier. Qualifier::from_str(q.as_str()) == Some(q)
    ///
    /// Rocq: `qualifier_parse_roundtrip` in SpaceFactory.v
    #[test]
    fn prop_qualifier_parse_roundtrip(qualifier in arb_qualifier()) {
        let s = qualifier.as_str();
        let parsed = Qualifier::from_str(s);
        prop_assert_eq!(parsed, Some(qualifier));
    }

    /// Property: All valid URNs can be parsed.
    ///
    /// ∀ urn ∈ all_valid_urns(). config_from_urn(urn) is Some
    ///
    /// Rocq: `all_valid_urns_parse_ok` in SpaceFactory.v
    #[test]
    fn prop_all_valid_urns_parse_ok(urn in arb_valid_urn()) {
        let config = config_from_urn(&urn);
        prop_assert!(
            config.is_some(),
            "Valid URN '{}' should parse to a config",
            urn
        );
    }

    /// Property: Invalid combinations fail gracefully.
    ///
    /// ∀ (inner, outer) where !is_valid_combination(inner, outer).
    ///   config_from_urn("rho:space:{inner}:{outer}:default") is None
    ///
    /// Rocq: `invalid_combos_fail_gracefully` in SpaceFactory.v
    #[test]
    fn prop_invalid_combos_fail_gracefully((inner, outer) in arb_invalid_combination()) {
        let urn = format!("rho:space:{}:{}:default", inner.as_str(), outer.as_str());
        let config = config_from_urn(&urn);
        prop_assert!(
            config.is_none(),
            "Invalid URN '{}' should not parse",
            urn
        );
    }
}

// =============================================================================
// Byte Name Mapping Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Same URN always produces the same byte name (determinism).
    ///
    /// ∀ urn ∈ valid_urns.
    ///   urn_to_byte_name(urn) == urn_to_byte_name(urn)
    ///
    /// Rocq: `byte_names_deterministic` in SpaceFactory.v
    #[test]
    fn prop_byte_names_deterministic(urn in arb_valid_urn()) {
        let byte1 = urn_to_byte_name(&urn);
        let byte2 = urn_to_byte_name(&urn);
        prop_assert_eq!(byte1, byte2, "Same URN should produce same byte");
    }

    /// Property: Byte name roundtrips back to the original URN.
    ///
    /// ∀ urn ∈ valid_urns.
    ///   byte_name_to_urn(urn_to_byte_name(urn)) == urn
    ///
    /// Rocq: `byte_name_roundtrip` in SpaceFactory.v
    #[test]
    fn prop_byte_name_roundtrip(urn in arb_valid_urn()) {
        if let Some(byte) = urn_to_byte_name(&urn) {
            let recovered = byte_name_to_urn(byte);
            prop_assert_eq!(
                recovered.as_ref(),
                Some(&urn),
                "Byte {} should roundtrip to URN {}",
                byte,
                urn
            );
        }
    }
}

// =============================================================================
// Configuration Computation Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Computed config has all fields populated.
    ///
    /// ∀ (inner, outer, qualifier) ∈ valid_combos.
    ///   compute_config(inner, _, outer, _, qualifier).outer == expected_outer
    ///   compute_config(inner, _, outer, _, qualifier).qualifier == expected_qualifier
    ///
    /// Rocq: `config_from_urn_complete` in SpaceFactory.v
    #[test]
    fn prop_config_from_urn_complete(
        (inner, outer) in arb_valid_combination(),
        qualifier in arb_qualifier()
    ) {
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

        let config = compute_config(inner, inner_params, outer, outer_params, qualifier);

        // Verify outer storage matches
        let expected_outer = match outer {
            OuterType::HashMap => OuterStorageType::HashMap,
            OuterType::PathMap => OuterStorageType::PathMap,
            OuterType::Array => OuterStorageType::Array { max_size: 1000, cyclic: false },
            OuterType::Vector => OuterStorageType::Vector,
            OuterType::HashSet => OuterStorageType::HashSet,
        };
        prop_assert_eq!(config.outer, expected_outer);

        // Verify qualifier matches
        prop_assert_eq!(config.qualifier, qualifier.to_space_qualifier());
    }

    /// Property: URN roundtrip preserves core config fields.
    ///
    /// ∀ urn ∈ valid_urns.
    ///   let config = config_from_urn(urn)
    ///   let recovered_urn = urn_from_config(config)
    ///   config_from_urn(recovered_urn).outer == config.outer
    ///   config_from_urn(recovered_urn).data_collection == config.data_collection
    ///
    /// Rocq: `urn_config_roundtrip` in SpaceFactory.v
    #[test]
    fn prop_urn_config_roundtrip(urn in arb_valid_urn()) {
        if let Some(config) = config_from_urn(&urn) {
            let recovered_urn = urn_from_config(&config);
            if let Some(recovered_config) = config_from_urn(&recovered_urn) {
                prop_assert_eq!(
                    config.outer, recovered_config.outer,
                    "Outer storage should roundtrip"
                );
                prop_assert_eq!(
                    config.data_collection, recovered_config.data_collection,
                    "Data collection should roundtrip"
                );
                prop_assert_eq!(
                    config.qualifier, recovered_config.qualifier,
                    "Qualifier should roundtrip"
                );
            }
        }
    }
}

// =============================================================================
// Inner/Outer Parameter Parsing Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: PriorityQueue parameter parsing extracts correct value.
    ///
    /// ∀ n ∈ 1..10.
    ///   parse_inner_with_params("priorityqueue({n})").priorities == n
    ///
    /// Rocq: `priorityqueue_param_parse` in SpaceFactory.v
    #[test]
    fn prop_priorityqueue_param_parse(n in 1usize..=100) {
        let s = format!("priorityqueue({})", n);
        let result = parse_inner_with_params(&s);
        prop_assert!(result.is_some());
        let (inner, params) = result.expect("should parse");
        prop_assert_eq!(inner, InnerType::PriorityQueue);
        match params {
            InnerParams::PriorityQueue { priorities } => {
                prop_assert_eq!(priorities, n);
            }
            _ => prop_assert!(false, "Expected PriorityQueue params"),
        }
    }

    /// Property: VectorDB parameter parsing extracts correct dimensions.
    ///
    /// ∀ d ∈ 2..512.
    ///   parse_inner_with_params("vectordb({d})").dimensions == d
    ///
    /// Rocq: `vectordb_param_parse` in SpaceFactory.v
    #[test]
    fn prop_vectordb_param_parse(d in 2usize..=512) {
        let s = format!("vectordb({})", d);
        let result = parse_inner_with_params(&s);
        prop_assert!(result.is_some());
        let (inner, params) = result.expect("should parse");
        prop_assert_eq!(inner, InnerType::VectorDB);
        match params {
            InnerParams::VectorDB { dimensions, backend } => {
                prop_assert_eq!(dimensions, d);
                prop_assert_eq!(backend, "rho");
            }
            _ => prop_assert!(false, "Expected VectorDB params"),
        }
    }

    /// Property: Array parameter parsing extracts size and cyclic flag.
    ///
    /// ∀ size ∈ 10..10000, cyclic ∈ bool.
    ///   parse_outer_with_params("array({size},{cyclic})") extracts correctly
    ///
    /// Rocq: `array_param_parse` in SpaceFactory.v
    #[test]
    fn prop_array_param_parse(size in 10usize..=10000, cyclic in any::<bool>()) {
        let s = format!("array({},{})", size, cyclic);
        let result = parse_outer_with_params(&s);
        prop_assert!(result.is_some());
        let (outer, params) = result.expect("should parse");
        prop_assert_eq!(outer, OuterType::Array);
        match params {
            OuterParams::Array { size: parsed_size, cyclic: parsed_cyclic } => {
                prop_assert_eq!(parsed_size, size);
                prop_assert_eq!(parsed_cyclic, cyclic);
            }
            _ => prop_assert!(false, "Expected Array params"),
        }
    }
}

// =============================================================================
// Theory Parsing Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: TheorySpec::parse roundtrips for builtin theories.
    ///
    /// ∀ name ∈ builtin_names.
    ///   TheorySpec::parse(name) == TheorySpec::Builtin(name)
    ///
    /// Rocq: `theory_builtin_parsed` in SpaceFactory.v
    #[test]
    fn prop_theory_builtin_parsed(name in arb_builtin_theory_name()) {
        let spec = TheorySpec::parse(&name);
        prop_assert_eq!(spec, TheorySpec::Builtin(name));
    }

    /// Property: TheorySpec::parse handles mettail: prefix.
    ///
    /// ∀ path. TheorySpec::parse("mettail:{path}") == TheorySpec::MeTTaILFile(path)
    ///
    /// Rocq: `theory_file_parsed` in SpaceFactory.v
    #[test]
    fn prop_theory_file_parsed(path in "[a-z]+/[a-z]+\\.metta") {
        let input = format!("mettail:{}", path);
        let spec = TheorySpec::parse(&input);
        prop_assert_eq!(spec, TheorySpec::MeTTaILFile(path));
    }

    /// Property: TheorySpec::parse handles inline: prefix.
    ///
    /// ∀ code. TheorySpec::parse("inline:{code}") == TheorySpec::InlineMeTTaIL(code)
    ///
    /// Rocq: `theory_inline_parsed` in SpaceFactory.v
    #[test]
    fn prop_theory_inline_parsed(code in "\\(: [A-Z][a-z]+ Type\\)") {
        let input = format!("inline:{}", code);
        let spec = TheorySpec::parse(&input);
        prop_assert_eq!(spec, TheorySpec::InlineMeTTaIL(code));
    }

    /// Property: URN with theory extension parses correctly.
    ///
    /// ∀ base_urn, theory_name.
    ///   parse_urn_with_theory("{base_urn}[theory={theory_name}]")
    ///     == (base_urn, Some(TheorySpec::Builtin(theory_name)))
    ///
    /// Rocq: `urn_with_theory_parsed` in SpaceFactory.v
    #[test]
    fn prop_urn_with_theory_parsed(
        urn in arb_valid_urn(),
        theory_name in arb_builtin_theory_name()
    ) {
        let full_urn = format!("{}[theory={}]", urn, theory_name);
        let (base, spec) = parse_urn_with_theory(&full_urn);

        prop_assert_eq!(base, urn);
        prop_assert_eq!(spec, Some(TheorySpec::Builtin(theory_name)));
    }

    /// Property: Builtin theory loader can load all advertised theories.
    ///
    /// ∀ name ∈ builtin_theories().
    ///   loader.load(TheorySpec::Builtin(name)).is_ok()
    ///
    /// Rocq: `builtin_loader_completeness` in SpaceFactory.v
    #[test]
    fn prop_builtin_loader_completeness(name in arb_builtin_theory_name()) {
        let loader = BuiltinTheoryLoader::new();
        let spec = TheorySpec::Builtin(name.clone());
        let result = loader.load(&spec);
        prop_assert!(
            result.is_ok(),
            "Builtin theory '{}' should load successfully",
            name
        );
    }
}

// =============================================================================
// Combination Validity Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Non-VectorDB inner types are valid with all outer types.
    ///
    /// ∀ inner ∈ {Bag, Queue, Stack, Set, Cell, PriorityQueue}, outer ∈ OuterType.
    ///   is_valid_combination(inner, outer) == true
    ///
    /// Rocq: `non_vectordb_always_valid` in SpaceFactory.v
    #[test]
    fn prop_non_vectordb_always_valid(
        inner in prop_oneof![
            Just(InnerType::Bag),
            Just(InnerType::Queue),
            Just(InnerType::Stack),
            Just(InnerType::Set),
            Just(InnerType::Cell),
            Just(InnerType::PriorityQueue),
        ],
        outer in arb_outer_type()
    ) {
        prop_assert!(
            is_valid_combination(inner, outer),
            "{:?} + {:?} should be valid",
            inner,
            outer
        );
    }

    /// Property: VectorDB is only valid with HashMap and Vector.
    ///
    /// is_valid_combination(VectorDB, HashMap) == true
    /// is_valid_combination(VectorDB, Vector) == true
    /// is_valid_combination(VectorDB, _) == false for others
    ///
    /// Rocq: `vectordb_outer_restrictions` in SpaceFactory.v
    #[test]
    fn prop_vectordb_outer_restrictions(outer in arb_outer_type()) {
        let expected = matches!(outer, OuterType::HashMap | OuterType::Vector);
        let actual = is_valid_combination(InnerType::VectorDB, outer);
        prop_assert_eq!(
            actual, expected,
            "VectorDB + {:?} validity should be {}",
            outer, expected
        );
    }
}

// =============================================================================
// Static Invariant Tests (Non-Proptest)
// =============================================================================

#[test]
fn test_valid_urn_count_is_96() {
    // 7 inner types × 5 outer types = 35 base combinations
    // VectorDB has 3 invalid outer types = 3 invalid
    // Valid combinations = 35 - 3 = 32
    // Each combination × 3 qualifiers = 32 × 3 = 96 valid URNs
    assert_eq!(valid_urn_count(), 96, "Expected exactly 96 valid URN combinations");
}

#[test]
fn test_all_valid_urns_unique() {
    let urns: Vec<String> = all_valid_urns().collect();
    let unique: HashSet<&String> = urns.iter().collect();
    assert_eq!(
        urns.len(),
        unique.len(),
        "All valid URNs should be unique"
    );
}

#[test]
fn test_byte_names_unique() {
    let mut seen_bytes: HashSet<u8> = HashSet::new();
    for urn in all_valid_urns() {
        if let Some(byte) = urn_to_byte_name(&urn) {
            assert!(
                seen_bytes.insert(byte),
                "Byte {} is duplicated for URN {}",
                byte,
                urn
            );
        }
    }
    // All 96 URNs should have unique bytes
    assert_eq!(seen_bytes.len(), 96, "All 96 URNs should have unique bytes");
}

#[test]
fn test_byte_name_range() {
    // Bytes should be in range 25-120 (SPACE_FACTORY_BASE_BYTE to BASE + 95)
    for urn in all_valid_urns() {
        if let Some(byte) = urn_to_byte_name(&urn) {
            assert!(
                (25..=120).contains(&byte),
                "Byte {} for URN {} should be in range 25-120",
                byte,
                urn
            );
        }
    }
}

#[test]
fn test_inner_type_all_constant() {
    // Verify InnerType::ALL contains all variants
    assert_eq!(InnerType::ALL.len(), 7);
    for inner in InnerType::ALL {
        assert!(InnerType::from_str(inner.as_str()).is_some());
    }
}

#[test]
fn test_outer_type_all_constant() {
    // Verify OuterType::ALL contains all variants
    assert_eq!(OuterType::ALL.len(), 5);
    for outer in OuterType::ALL {
        assert!(OuterType::from_str(outer.as_str()).is_some());
    }
}

#[test]
fn test_qualifier_all_constant() {
    // Verify Qualifier::ALL contains all variants
    assert_eq!(Qualifier::ALL.len(), 3);
    for q in Qualifier::ALL {
        assert!(Qualifier::from_str(q.as_str()).is_some());
    }
}

#[test]
fn test_theory_spec_display_roundtrip() {
    // Verify TheorySpec Display format matches parse expectations
    let cases = vec![
        TheorySpec::Builtin("Nat".to_string()),
        TheorySpec::MeTTaILFile("types/nat.metta".to_string()),
        TheorySpec::InlineMeTTaIL("(: Nat Type)".to_string()),
        TheorySpec::Uri("https://example.com/nat".to_string()),
    ];

    for spec in cases {
        let display = spec.to_string();
        // Builtin doesn't have prefix, others do
        match &spec {
            TheorySpec::Builtin(name) => assert_eq!(display, *name),
            TheorySpec::MeTTaILFile(path) => assert_eq!(display, format!("mettail:{}", path)),
            TheorySpec::InlineMeTTaIL(code) => assert_eq!(display, format!("inline:{}", code)),
            TheorySpec::Uri(uri) => assert_eq!(display, format!("uri:{}", uri)),
        }
    }
}

#[test]
fn test_config_from_full_urn_with_valid_theory() {
    let loader = BuiltinTheoryLoader::new();
    let result = config_from_full_urn(
        "rho:space:bag:hashmap:default[theory=Nat]",
        &loader,
    );
    assert!(result.is_ok());
    let config = result.expect("config should parse");
    assert!(config.theory.is_some());
    assert_eq!(config.theory.as_ref().expect("theory").name(), "Nat");
}

#[test]
fn test_config_from_full_urn_without_theory() {
    let loader = BuiltinTheoryLoader::new();
    let result = config_from_full_urn("rho:space:queue:hashmap:default", &loader);
    assert!(result.is_ok());
    let config = result.expect("config should parse");
    assert!(config.theory.is_none());
}

#[test]
fn test_config_from_full_urn_unknown_theory() {
    let loader = BuiltinTheoryLoader::new();
    let result = config_from_full_urn(
        "rho:space:bag:hashmap:default[theory=UnknownTheory]",
        &loader,
    );
    assert!(result.is_err());
}
