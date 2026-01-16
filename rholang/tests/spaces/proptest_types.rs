//! Property-Based Tests for Reified RSpaces Types Module
//!
//! This module provides comprehensive property-based testing for the core types
//! used in the Reified RSpaces implementation.
//!
//! # Formal Correspondence
//! - `Substitution.v`: Theory validation properties
//! - `PathMapQuantale.v`: Path prefix/suffix properties
//! - `Safety/Properties.v`: Qualifier safety guarantees
//!
//! # Test Coverage
//! - Theory validation (NullTheory accepts all, SimpleTypeTheory validates)
//! - SpaceQualifier properties (mobility, persistence, concurrency)
//! - SpaceId hex roundtrip
//! - Path prefix/suffix operations
//! - AggregatedDatum semantics
//! - SpaceConfig builder patterns
//! - GasConfiguration properties
//! - Par-to-path conversion roundtrips
//!
//! # Rholang Syntax Examples
//!
//! ```rholang
//! // Theory-validated space
//! new NatSpace(`rho:space:bag:hashmap:default[theory=Nat]`) in {
//!   NatSpace!({}, *space) |
//!   use space {
//!     ch!(42)    // Validates as Nat
//!     ch!(-5)    // Would fail: Int not allowed
//!   }
//! }
//!
//! // PathMap with prefix semantics
//! new space(`rho:space:bag:pathmap:default`) in {
//!   @["sys", "auth"]!(credentials) |
//!   for (data <- @["sys"]) { ... }  // Receives from all @["sys", ...] paths
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::{
    // Theory types
    Theory, NullTheory, SimpleTypeTheory,
    // Space types
    SpaceQualifier, SpaceId, SpaceConfig, InnerCollectionType, OuterStorageType,
    GasConfiguration,
    // PathMap types
    AggregatedDatum, get_path_suffix, path_prefixes, is_path_prefix,
    path_element_boundaries, par_to_path, path_to_par, is_par_path,
};

use models::rhoapi::{expr::ExprInstance, EList, Expr, Par};

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 500;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Arbitrary Generators
// =============================================================================

/// Generate an arbitrary term string.
fn arb_term() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_() ]{0,100}"
}

/// Generate an arbitrary type name.
fn arb_type_name() -> impl Strategy<Value = String> {
    "[A-Z][a-zA-Z0-9]{0,20}"
}

/// Generate a list of type names.
fn arb_type_names(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop_vec(arb_type_name(), min..=max)
}

/// Generate an arbitrary SpaceQualifier.
fn arb_qualifier() -> impl Strategy<Value = SpaceQualifier> {
    prop_oneof![
        Just(SpaceQualifier::Default),
        Just(SpaceQualifier::Temp),
        Just(SpaceQualifier::Seq),
    ]
}

/// Generate arbitrary hex bytes for SpaceId.
fn arb_hex_bytes() -> impl Strategy<Value = Vec<u8>> {
    prop_vec(any::<u8>(), 1..=64)
}

/// Generate a path as a byte vector.
fn arb_path_bytes() -> impl Strategy<Value = Vec<u8>> {
    prop_vec(any::<u8>(), 0..=20)
}

/// Generate a gas limit value.
fn arb_gas_limit() -> impl Strategy<Value = u64> {
    any::<u64>()
}

/// Generate a cost multiplier value.
fn arb_multiplier() -> impl Strategy<Value = f64> {
    0.1f64..10.0f64
}

/// Generate an integer for path elements.
fn arb_path_int() -> impl Strategy<Value = i64> {
    any::<i64>()
}

/// Generate a string for path elements.
fn arb_path_string() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_]{1,50}"
}

// =============================================================================
// Theory Validation Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: NullTheory accepts all data.
    ///
    /// ∀ term. NullTheory.validate(term) == Ok(())
    ///
    /// Rocq: `null_theory_accepts_all` in Substitution.v
    #[test]
    fn prop_null_theory_accepts_all(term in arb_term()) {
        let theory = NullTheory;
        prop_assert!(
            theory.validate(&term).is_ok(),
            "NullTheory should accept all terms"
        );
    }

    /// Property: NullTheory has correct name.
    #[test]
    fn prop_null_theory_name(_seed in any::<u64>()) {
        let theory = NullTheory;
        prop_assert_eq!(theory.name(), "NullTheory");
    }

    /// Property: SimpleTypeTheory accepts terms starting with allowed types.
    ///
    /// ∀ type ∈ allowed_types, suffix. SimpleTypeTheory.validate(type + suffix) == Ok(())
    ///
    /// Rocq: `simple_theory_type_check` in Substitution.v
    #[test]
    fn prop_simple_theory_accepts_allowed_type(
        type_name in arb_type_name(),
        suffix in "[a-z0-9 ]{0,30}"
    ) {
        let theory = SimpleTypeTheory::new("TestTheory", vec![type_name.clone()]);
        let term = format!("{}{}", type_name, suffix);
        prop_assert!(
            theory.validate(&term).is_ok(),
            "SimpleTypeTheory should accept term starting with allowed type"
        );
    }

    /// Property: SimpleTypeTheory rejects terms not starting with allowed types.
    ///
    /// ∀ type ∉ allowed_types. SimpleTypeTheory.validate(type + ...) == Err(...)
    ///
    /// Rocq: `simple_theory_rejects_invalid` in Substitution.v
    #[test]
    fn prop_simple_theory_rejects_invalid(
        allowed in arb_type_name(),
        invalid_prefix in "[a-z]{1,5}"  // lowercase, won't match uppercase type names
    ) {
        let theory = SimpleTypeTheory::new("TestTheory", vec![allowed.clone()]);
        let term = format!("{}SomeValue", invalid_prefix);

        // Only reject if the invalid prefix is definitely not allowed
        if !term.starts_with(&allowed) {
            prop_assert!(
                theory.validate(&term).is_err(),
                "SimpleTypeTheory should reject terms not starting with allowed types"
            );
        }
    }

    /// Property: SimpleTypeTheory has_type returns true for allowed types.
    ///
    /// Rocq: `simple_theory_has_type` in Substitution.v
    #[test]
    fn prop_simple_theory_has_type(types in arb_type_names(1, 5)) {
        let theory = SimpleTypeTheory::new("TestTheory", types.clone());

        for t in &types {
            prop_assert!(
                theory.has_type(t),
                "has_type should return true for allowed types"
            );
        }
    }

    /// Property: Theory clone_box preserves behavior.
    ///
    /// Rocq: `theory_clone_preserves` in Substitution.v
    #[test]
    fn prop_theory_clone_preserves_behavior(
        types in arb_type_names(1, 3),
        term in arb_term()
    ) {
        let theory = SimpleTypeTheory::new("TestTheory", types);
        let cloned = theory.clone_box();

        prop_assert_eq!(
            theory.validate(&term).is_ok(),
            cloned.validate(&term).is_ok(),
            "Cloned theory should have same validation behavior"
        );
    }
}

// =============================================================================
// SpaceQualifier Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Only Seq qualifier is non-mobile.
    ///
    /// ∀ q. q.is_mobile() == (q ≠ Seq)
    ///
    /// Rocq: `qualifier_mobility` in Safety/Properties.v
    #[test]
    fn prop_qualifier_mobility(q in arb_qualifier()) {
        let expected = !matches!(q, SpaceQualifier::Seq);
        prop_assert_eq!(
            q.is_mobile(),
            expected,
            "Only Seq should be non-mobile"
        );
    }

    /// Property: Only Default qualifier is persistent.
    ///
    /// ∀ q. q.is_persistent() == (q == Default)
    ///
    /// Rocq: `qualifier_persistence` in Safety/Properties.v
    #[test]
    fn prop_qualifier_persistence(q in arb_qualifier()) {
        let expected = matches!(q, SpaceQualifier::Default);
        prop_assert_eq!(
            q.is_persistent(),
            expected,
            "Only Default should be persistent"
        );
    }

    /// Property: Only Seq qualifier is non-concurrent.
    ///
    /// ∀ q. q.is_concurrent() == (q ≠ Seq)
    ///
    /// Rocq: `qualifier_concurrency` in Safety/Properties.v
    #[test]
    fn prop_qualifier_concurrency(q in arb_qualifier()) {
        let expected = !matches!(q, SpaceQualifier::Seq);
        prop_assert_eq!(
            q.is_concurrent(),
            expected,
            "Only Seq should be non-concurrent"
        );
    }

    /// Property: Qualifier Display is consistent.
    #[test]
    fn prop_qualifier_display(q in arb_qualifier()) {
        let display = format!("{}", q);
        let expected = match q {
            SpaceQualifier::Default => "default",
            SpaceQualifier::Temp => "temp",
            SpaceQualifier::Seq => "seq",
        };
        prop_assert_eq!(display, expected);
    }
}

// =============================================================================
// SpaceId Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: SpaceId hex encoding roundtrips.
    ///
    /// ∀ bytes. SpaceId::from_hex(SpaceId(bytes).to_hex()) == SpaceId(bytes)
    ///
    /// Rocq: `space_id_hex_roundtrip` in Types.v
    #[test]
    fn prop_space_id_hex_roundtrip(bytes in arb_hex_bytes()) {
        let original = SpaceId::new(bytes.clone());
        let hex = original.to_hex();
        let recovered = SpaceId::from_hex(&hex).expect("valid hex");

        prop_assert_eq!(
            original.as_bytes(),
            recovered.as_bytes(),
            "SpaceId hex encoding should roundtrip"
        );
    }

    /// Property: SpaceId default_space has 32 zero bytes.
    #[test]
    fn prop_space_id_default(_seed in any::<u64>()) {
        let id = SpaceId::default_space();
        prop_assert_eq!(id.as_bytes().len(), 32);
        prop_assert!(id.as_bytes().iter().all(|&b| b == 0));
    }

    /// Property: SpaceId to_path returns same bytes as as_bytes.
    #[test]
    fn prop_space_id_path_bytes(bytes in arb_hex_bytes()) {
        let id = SpaceId::new(bytes);
        prop_assert_eq!(id.to_path(), id.as_bytes());
    }
}

// =============================================================================
// Path Prefix/Suffix Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Empty prefix yields entire path as suffix.
    ///
    /// ∀ path. get_path_suffix([], path) == Some(path)
    ///
    /// Rocq: `empty_prefix_full_suffix` in PathMapQuantale.v
    #[test]
    fn prop_empty_prefix_full_suffix(path in arb_path_bytes()) {
        let suffix = get_path_suffix(&[], &path);
        prop_assert_eq!(suffix, Some(path));
    }

    /// Property: Path is prefix of itself with empty suffix.
    ///
    /// ∀ path. get_path_suffix(path, path) == Some([])
    ///
    /// Rocq: `self_prefix_empty_suffix` in PathMapQuantale.v
    #[test]
    fn prop_self_prefix_empty_suffix(path in arb_path_bytes()) {
        let suffix = get_path_suffix(&path, &path);
        prop_assert_eq!(suffix, Some(vec![]));
    }

    /// Property: get_path_suffix returns None for non-prefixes.
    ///
    /// ∀ prefix, child where !child.starts_with(prefix). get_path_suffix(prefix, child) == None
    ///
    /// Rocq: `non_prefix_returns_none` in PathMapQuantale.v
    #[test]
    fn prop_non_prefix_returns_none(
        prefix in prop_vec(1u8..10u8, 2..=5),
        child in prop_vec(10u8..20u8, 2..=5)
    ) {
        // These should not be prefix-related due to different byte ranges
        if !child.starts_with(&prefix) {
            prop_assert_eq!(
                get_path_suffix(&prefix, &child),
                None,
                "Non-prefix should return None"
            );
        }
    }

    /// Property: is_path_prefix is consistent with get_path_suffix.
    ///
    /// ∀ prefix, path. is_path_prefix(prefix, path) == get_path_suffix(prefix, path).is_some()
    ///
    /// Rocq: `is_prefix_consistent` in PathMapQuantale.v
    #[test]
    fn prop_is_path_prefix_consistent(
        prefix in arb_path_bytes(),
        path in arb_path_bytes()
    ) {
        let is_prefix = is_path_prefix(&prefix, &path);
        let has_suffix = get_path_suffix(&prefix, &path).is_some();
        prop_assert_eq!(
            is_prefix,
            has_suffix,
            "is_path_prefix should be consistent with get_path_suffix"
        );
    }

    /// Property: path_prefixes generates increasing lengths.
    ///
    /// ∀ path. prefixes = path_prefixes(path) ⟹ prefixes[i].len() == i + 1
    ///
    /// Rocq: `prefixes_increasing_length` in PathMapQuantale.v
    #[test]
    fn prop_path_prefixes_increasing_length(path in prop_vec(any::<u8>(), 1..=10)) {
        let prefixes = path_prefixes(&path);
        prop_assert_eq!(prefixes.len(), path.len());

        for (i, prefix) in prefixes.iter().enumerate() {
            prop_assert_eq!(
                prefix.len(),
                i + 1,
                "Prefix at index {} should have length {}", i, i + 1
            );
        }
    }

    /// Property: All prefixes from path_prefixes are actual prefixes.
    ///
    /// ∀ path, p ∈ path_prefixes(path). is_path_prefix(p, path)
    ///
    /// Rocq: `all_prefixes_valid` in PathMapQuantale.v
    #[test]
    fn prop_all_prefixes_valid(path in prop_vec(any::<u8>(), 1..=10)) {
        let prefixes = path_prefixes(&path);

        for prefix in prefixes {
            prop_assert!(
                is_path_prefix(&prefix, &path),
                "All prefixes should be valid prefixes of the original path"
            );
        }
    }
}

// =============================================================================
// AggregatedDatum Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: AggregatedDatum::exact has empty suffix.
    ///
    /// ∀ data, persist. AggregatedDatum::exact(data, persist).is_exact_match()
    ///
    /// Rocq: `exact_has_empty_suffix` in PathMapStore.v
    #[test]
    fn prop_aggregated_exact_has_empty_suffix(data in any::<i32>(), persist in any::<bool>()) {
        let datum = AggregatedDatum::exact(data, persist);
        prop_assert!(datum.is_exact_match());
        prop_assert!(datum.suffix_key.is_empty());
        prop_assert_eq!(datum.suffix_depth(), 0);
    }

    /// Property: AggregatedDatum::new with non-empty suffix is not exact.
    ///
    /// ∀ suffix ≠ [], data. !AggregatedDatum::new(suffix, data, _).is_exact_match()
    ///
    /// Rocq: `non_empty_suffix_not_exact` in PathMapStore.v
    #[test]
    fn prop_non_empty_suffix_not_exact(
        suffix in prop_vec(any::<u8>(), 1..=5),
        data in any::<i32>()
    ) {
        let datum = AggregatedDatum::new(suffix.clone(), data, false);
        prop_assert!(!datum.is_exact_match());
        prop_assert_eq!(datum.suffix_depth(), suffix.len());
    }

    /// Property: AggregatedDatum::map preserves suffix and persist.
    ///
    /// ∀ datum, f. datum.map(f).suffix_key == datum.suffix_key ∧ datum.map(f).persist == datum.persist
    ///
    /// Rocq: `map_preserves_metadata` in PathMapStore.v
    #[test]
    fn prop_aggregated_map_preserves_metadata(
        suffix in prop_vec(any::<u8>(), 0..=5),
        data in any::<i32>(),
        persist in any::<bool>()
    ) {
        let datum = AggregatedDatum::new(suffix.clone(), data, persist);
        let mapped = datum.map(|x| x.wrapping_add(1));

        prop_assert_eq!(mapped.suffix_key, suffix);
        prop_assert_eq!(mapped.persist, persist);
        prop_assert_eq!(mapped.data, data.wrapping_add(1));
    }

    /// Property: AggregatedDatum clone equals original.
    #[test]
    fn prop_aggregated_clone_equals(
        suffix in prop_vec(any::<u8>(), 0..=5),
        data in any::<i32>(),
        persist in any::<bool>()
    ) {
        let datum = AggregatedDatum::new(suffix, data, persist);
        let cloned = datum.clone();

        prop_assert_eq!(datum, cloned);
    }
}

// =============================================================================
// SpaceConfig Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: SpaceConfig default has expected values.
    #[test]
    fn prop_space_config_default(_seed in any::<u64>()) {
        let config = SpaceConfig::default();
        prop_assert_eq!(&config.outer, &OuterStorageType::PathMap);
        prop_assert_eq!(&config.data_collection, &InnerCollectionType::Bag);
        prop_assert_eq!(config.qualifier, SpaceQualifier::Default);
        prop_assert!(config.theory.is_none());
        prop_assert!(config.has_gas()); // Gas enabled by default
    }

    /// Property: SpaceConfig with_qualifier preserves qualifier.
    #[test]
    fn prop_space_config_with_qualifier(q in arb_qualifier()) {
        let config = SpaceConfig::default().with_qualifier(q);
        prop_assert_eq!(config.qualifier, q);
    }

    /// Property: SpaceConfig with_theory makes has_theory true.
    #[test]
    fn prop_space_config_with_theory(_seed in any::<u64>()) {
        let theory = NullTheory;
        let config = SpaceConfig::default().with_theory(Box::new(theory));
        prop_assert!(config.has_theory());
        prop_assert_eq!(config.theory_name(), Some("NullTheory"));
    }

    /// Property: SpaceConfig without_theory clears theory.
    #[test]
    fn prop_space_config_without_theory(_seed in any::<u64>()) {
        let config = SpaceConfig::default()
            .with_theory(Box::new(NullTheory))
            .without_theory();

        prop_assert!(!config.has_theory());
        prop_assert_eq!(config.theory_name(), None);
    }

    /// Property: SpaceConfig without theory validates all data.
    #[test]
    fn prop_space_config_no_theory_validates_all(term in arb_term()) {
        let config = SpaceConfig::default();
        prop_assert!(config.validate_data(&term).is_ok());
    }

    /// Property: SpaceConfig is_persistent consistent with qualifier.
    #[test]
    fn prop_space_config_persistence(q in arb_qualifier()) {
        let config = SpaceConfig::default().with_qualifier(q);
        prop_assert_eq!(config.is_persistent(), q.is_persistent());
    }

    /// Property: SpaceConfig is_concurrent consistent with qualifier.
    #[test]
    fn prop_space_config_concurrency(q in arb_qualifier()) {
        let config = SpaceConfig::default().with_qualifier(q);
        prop_assert_eq!(config.is_concurrent(), q.is_concurrent());
    }

    /// Property: SpaceConfig is_mobile consistent with qualifier.
    #[test]
    fn prop_space_config_mobility(q in arb_qualifier()) {
        let config = SpaceConfig::default().with_qualifier(q);
        prop_assert_eq!(config.is_mobile(), q.is_mobile());
    }
}

// =============================================================================
// GasConfiguration Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: GasConfiguration::default has enabled gas.
    #[test]
    fn prop_gas_config_default_enabled(_seed in any::<u64>()) {
        let gas = GasConfiguration::default();
        prop_assert!(gas.enabled);
        prop_assert!(gas.initial_limit > 0);
    }

    /// Property: GasConfiguration::disabled has disabled gas.
    #[test]
    fn prop_gas_config_disabled(_seed in any::<u64>()) {
        let gas = GasConfiguration::disabled();
        prop_assert!(!gas.enabled);
    }

    /// Property: GasConfiguration::unlimited has max limit.
    #[test]
    fn prop_gas_config_unlimited(_seed in any::<u64>()) {
        let gas = GasConfiguration::unlimited();
        prop_assert!(gas.enabled);
        prop_assert_eq!(gas.initial_limit, u64::MAX);
    }

    /// Property: GasConfiguration::with_limit preserves limit.
    #[test]
    fn prop_gas_config_with_limit(limit in arb_gas_limit()) {
        let gas = GasConfiguration::with_limit(limit);
        prop_assert_eq!(gas.initial_limit, limit);
        prop_assert!(gas.enabled);
    }

    /// Property: GasConfiguration::with_multiplier preserves multiplier.
    #[test]
    fn prop_gas_config_with_multiplier(mult in arb_multiplier()) {
        let gas = GasConfiguration::default().with_multiplier(mult);
        prop_assert!((gas.cost_multiplier - mult).abs() < 0.0001);
    }

    /// Property: SpaceConfig with_gas_limit sets gas correctly.
    #[test]
    fn prop_space_config_gas_limit(limit in arb_gas_limit()) {
        let config = SpaceConfig::default().with_gas_limit(limit);
        prop_assert!(config.has_gas());
        prop_assert_eq!(config.gas_limit(), limit);
    }

    /// Property: SpaceConfig with_disabled_gas disables gas.
    #[test]
    fn prop_space_config_disabled_gas(_seed in any::<u64>()) {
        let config = SpaceConfig::default()
            .with_gas_limit(1000)
            .with_disabled_gas();

        prop_assert!(!config.has_gas());
    }
}

// =============================================================================
// Par-to-Path Conversion Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: Integer path roundtrips through par_to_path/path_to_par.
    ///
    /// ∀ n. path_to_par(par_to_path(@[n])) contains GInt(n)
    ///
    /// Rocq: `integer_path_roundtrip` in PathMapStore.v
    #[test]
    fn prop_integer_path_roundtrip(n in arb_path_int()) {
        // Create @[n]
        let par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(n)),
                }])],
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);

        let path = par_to_path(&par);
        prop_assert!(path.is_some(), "par_to_path should succeed for integer list");

        let path = path.expect("path should exist");
        let recovered = path_to_par(&path);

        // Extract the integer from recovered
        if let Some(ExprInstance::EListBody(elist)) = &recovered.exprs.first().and_then(|e| e.expr_instance.as_ref()) {
            prop_assert_eq!(elist.ps.len(), 1);
            if let Some(ExprInstance::GInt(recovered_n)) = &elist.ps[0].exprs.first().and_then(|e| e.expr_instance.as_ref()) {
                prop_assert_eq!(*recovered_n, n, "Integer should roundtrip");
            } else {
                prop_assert!(false, "Expected GInt in recovered path");
            }
        } else {
            prop_assert!(false, "Expected EListBody in recovered path");
        }
    }

    /// Property: String path roundtrips through par_to_path/path_to_par.
    ///
    /// ∀ s. path_to_par(par_to_path(@[s])) contains GString(s)
    ///
    /// Rocq: `string_path_roundtrip` in PathMapStore.v
    #[test]
    fn prop_string_path_roundtrip(s in arb_path_string()) {
        // Create @[s]
        let par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GString(s.clone())),
                }])],
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);

        let path = par_to_path(&par);
        prop_assert!(path.is_some(), "par_to_path should succeed for string list");

        let path = path.expect("path should exist");
        let recovered = path_to_par(&path);

        // Extract the string from recovered
        if let Some(ExprInstance::EListBody(elist)) = &recovered.exprs.first().and_then(|e| e.expr_instance.as_ref()) {
            prop_assert_eq!(elist.ps.len(), 1);
            if let Some(ExprInstance::GString(recovered_s)) = &elist.ps[0].exprs.first().and_then(|e| e.expr_instance.as_ref()) {
                prop_assert_eq!(recovered_s.clone(), s, "String should roundtrip");
            } else {
                prop_assert!(false, "Expected GString in recovered path");
            }
        } else {
            prop_assert!(false, "Expected EListBody in recovered path");
        }
    }

    /// Property: is_par_path returns true for valid path pars.
    ///
    /// Rocq: `is_par_path_valid` in PathMapStore.v
    #[test]
    fn prop_is_par_path_valid(n in arb_path_int()) {
        let par = Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EListBody(EList {
                ps: vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GInt(n)),
                }])],
                locally_free: vec![],
                connective_used: false,
                remainder: None,
            })),
        }]);

        prop_assert!(is_par_path(&par), "Valid path par should return true");
    }

    /// Property: Empty path roundtrips correctly.
    #[test]
    fn prop_empty_path_roundtrip(_seed in any::<u64>()) {
        let empty_path: Vec<u8> = vec![];
        let par = path_to_par(&empty_path);

        // Should produce empty EList
        if let Some(ExprInstance::EListBody(elist)) = &par.exprs.first().and_then(|e| e.expr_instance.as_ref()) {
            prop_assert!(elist.ps.is_empty(), "Empty path should produce empty EList");
        } else {
            prop_assert!(false, "Expected EListBody for empty path");
        }
    }
}

// =============================================================================
// InnerCollectionType and OuterStorageType Properties
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: InnerCollectionType default is Bag.
    #[test]
    fn prop_inner_collection_default(_seed in any::<u64>()) {
        prop_assert_eq!(InnerCollectionType::default(), InnerCollectionType::Bag);
    }

    /// Property: OuterStorageType default is PathMap.
    #[test]
    fn prop_outer_storage_default(_seed in any::<u64>()) {
        prop_assert_eq!(OuterStorageType::default(), OuterStorageType::PathMap);
    }

    /// Property: InnerCollectionType Display is non-empty.
    #[test]
    fn prop_inner_collection_display(_seed in any::<u64>()) {
        let types = vec![
            InnerCollectionType::Bag,
            InnerCollectionType::Queue,
            InnerCollectionType::Stack,
            InnerCollectionType::Set,
            InnerCollectionType::Cell,
            InnerCollectionType::PriorityQueue { priorities: 3 },
            InnerCollectionType::VectorDB { dimensions: 128, backend: "rho".to_string() },
        ];

        for t in types {
            let display = format!("{}", t);
            prop_assert!(!display.is_empty(), "Display should be non-empty");
        }
    }

    /// Property: OuterStorageType Display is non-empty.
    #[test]
    fn prop_outer_storage_display(_seed in any::<u64>()) {
        let types = vec![
            OuterStorageType::HashMap,
            OuterStorageType::PathMap,
            OuterStorageType::Vector,
            OuterStorageType::HashSet,
            OuterStorageType::Array { max_size: 100, cyclic: false },
        ];

        for t in types {
            let display = format!("{}", t);
            prop_assert!(!display.is_empty(), "Display should be non-empty");
        }
    }
}

// =============================================================================
// Static Invariant Tests (Non-Proptest)
// =============================================================================

#[test]
fn test_null_theory_description() {
    let theory = NullTheory;
    assert_eq!(theory.description(), "Accepts all data without validation");
}

#[test]
fn test_simple_type_theory_description() {
    let theory = SimpleTypeTheory::new("Test", vec!["Int".to_string()]);
    assert_eq!(theory.description(), "Simple type validation against allowed type names");
}

#[test]
fn test_space_config_hashmap_bag() {
    let config = SpaceConfig::hashmap_bag();
    assert_eq!(config.outer, OuterStorageType::HashMap);
    assert_eq!(config.data_collection, InnerCollectionType::Bag);
}

#[test]
fn test_space_config_queue() {
    let config = SpaceConfig::queue();
    assert_eq!(config.data_collection, InnerCollectionType::Queue);
    assert_eq!(config.continuation_collection, InnerCollectionType::Queue);
}

#[test]
fn test_space_config_stack() {
    let config = SpaceConfig::stack();
    assert_eq!(config.data_collection, InnerCollectionType::Stack);
}

#[test]
fn test_space_config_set() {
    let config = SpaceConfig::set();
    assert_eq!(config.data_collection, InnerCollectionType::Set);
}

#[test]
fn test_space_config_cell() {
    let config = SpaceConfig::cell();
    assert_eq!(config.data_collection, InnerCollectionType::Cell);
}

#[test]
fn test_space_config_vector_db() {
    let config = SpaceConfig::vector_db(128);
    match config.data_collection {
        InnerCollectionType::VectorDB { dimensions, backend } => {
            assert_eq!(dimensions, 128);
            assert_eq!(backend, "rho");
        }
        _ => panic!("Expected VectorDB"),
    }
}

#[test]
fn test_space_config_seq() {
    let config = SpaceConfig::seq();
    assert_eq!(config.qualifier, SpaceQualifier::Seq);
    assert!(!config.is_mobile());
    assert!(!config.is_concurrent());
    assert!(!config.is_persistent());
}

#[test]
fn test_space_config_temp() {
    let config = SpaceConfig::temp();
    assert_eq!(config.qualifier, SpaceQualifier::Temp);
    assert!(config.is_mobile());
    assert!(config.is_concurrent());
    assert!(!config.is_persistent());
}

#[test]
fn test_path_element_boundaries_empty() {
    let boundaries = path_element_boundaries(&[]);
    assert!(boundaries.is_empty());
}

#[test]
fn test_path_element_boundaries_integer() {
    // Single integer: tag (0x01) + 8 bytes = 9 bytes
    let path = vec![0x01, 42, 0, 0, 0, 0, 0, 0, 0];
    let boundaries = path_element_boundaries(&path);
    assert_eq!(boundaries, vec![9]);
}

#[test]
fn test_path_element_boundaries_string() {
    // Single string "hi": tag (0x02) + varint(2) + "hi" = 4 bytes
    let path = vec![0x02, 2, b'h', b'i'];
    let boundaries = path_element_boundaries(&path);
    assert_eq!(boundaries, vec![4]);
}

#[test]
fn test_space_id_display() {
    let id = SpaceId::new(vec![0xde, 0xad, 0xbe, 0xef]);
    let display = format!("{}", id);
    assert!(display.contains("deadbeef"));
}
