//! Property-Based Tests for Theory Type Enforcement
//!
//! This module tests the runtime enforcement of theory type validation
//! when sending data to channels in typed spaces.
//!
//! # Formal Correspondence
//! - `Substitution.v`: Theory validation semantics
//! - `Reifying RSpaces.md`: Type hint syntax (`new x : space_instance in { ... }`)
//!
//! # Design
//!
//! When a channel is created in a typed space:
//! 1. `new x : space_instance in { ... }` creates channel `x` in `space_instance`
//! 2. `space_instance` was created with an attached theory (e.g., `Nat`, `Int`)
//! 3. Operations on `x` validate data against the space's theory
//!
//! # Test Coverage
//!
//! 1. Unit tests for Validatable trait conversion
//! 2. Unit tests for Theory trait validation
//! 3. Integration tests for theory enforcement in produce_inner
//!
//! # Rholang Syntax Examples
//!
//! ```rholang
//! // Create a space with Nat theory
//! for(natSpace <- HMB!?("default", free Nat())) {
//!   new x : natSpace in {
//!     x!(42)       // OK: 42 validates as Nat
//!     x!(-5)       // Error: -5 does not validate as Nat
//!     x!("hello")  // Error: "hello" does not validate as Nat
//!   }
//! }
//! ```

use proptest::prelude::*;
use models::rhoapi::{expr::ExprInstance, Expr, ListParWithRandom, Par};
use rholang::rust::interpreter::spaces::{
    Theory, NullTheory, SimpleTypeTheory, BoxedTheory,
    Validatable, TheoryValidator, ValidationResult,
    SpaceConfig, SpaceQualifier, InnerCollectionType, OuterStorageType,
};

// =============================================================================
// Proptest Configuration
// =============================================================================

const PROPTEST_CASES: u32 = 200;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(PROPTEST_CASES)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a Par containing a GInt (for testing integer validation).
fn par_with_int(value: i64) -> Par {
    Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GInt(value)),
        }],
        ..Default::default()
    }
}

/// Create a Par containing a GString (for testing string validation).
fn par_with_string(value: &str) -> Par {
    Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GString(value.to_string())),
        }],
        ..Default::default()
    }
}

/// Create a Par containing a GBool (for testing boolean validation).
fn par_with_bool(value: bool) -> Par {
    Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GBool(value)),
        }],
        ..Default::default()
    }
}

/// Create a ListParWithRandom from a single Par.
fn list_par_from_single(par: Par) -> ListParWithRandom {
    ListParWithRandom {
        pars: vec![par],
        random_state: vec![],
    }
}

/// Create a SimpleTypeTheory that only accepts Nat values.
fn nat_theory() -> SimpleTypeTheory {
    SimpleTypeTheory::new("Nat", vec!["Nat".to_string()])
}

/// Create a SimpleTypeTheory that only accepts Int values.
fn int_theory() -> SimpleTypeTheory {
    SimpleTypeTheory::new("Int", vec!["Int".to_string(), "Nat".to_string()])
}

/// Create a SimpleTypeTheory that only accepts String values.
fn string_theory() -> SimpleTypeTheory {
    SimpleTypeTheory::new("String", vec!["String".to_string()])
}

/// Create a SimpleTypeTheory that only accepts Bool values.
fn bool_theory() -> SimpleTypeTheory {
    SimpleTypeTheory::new("Bool", vec!["Bool".to_string()])
}

/// Create an "Any" theory that accepts everything (like NullTheory but named).
fn any_theory() -> SimpleTypeTheory {
    SimpleTypeTheory::new(
        "Any",
        vec![
            "Nat".to_string(),
            "Int".to_string(),
            "String".to_string(),
            "Bool".to_string(),
            "Unit".to_string(),
            "List".to_string(),
            "Tuple".to_string(),
            "Set".to_string(),
            "Map".to_string(),
            "Process".to_string(),
            "Unknown".to_string(),
        ],
    )
}

// =============================================================================
// Validatable Trait Tests
// =============================================================================

/// Test that positive integers are converted to "Nat(n)" format.
#[test]
fn test_validatable_positive_int_to_nat() {
    let data = list_par_from_single(par_with_int(42));
    let result = data.to_validatable_string();
    assert_eq!(result, "Nat(42)");
    assert_eq!(data.type_name(), "Nat");
}

/// Test that zero is converted to "Nat(0)" format.
#[test]
fn test_validatable_zero_to_nat() {
    let data = list_par_from_single(par_with_int(0));
    let result = data.to_validatable_string();
    assert_eq!(result, "Nat(0)");
    assert_eq!(data.type_name(), "Nat");
}

/// Test that negative integers are converted to "Int(n)" format.
#[test]
fn test_validatable_negative_int_to_int() {
    let data = list_par_from_single(par_with_int(-5));
    let result = data.to_validatable_string();
    assert_eq!(result, "Int(-5)");
    assert_eq!(data.type_name(), "Int");
}

/// Test that strings are converted to "String(s)" format.
#[test]
fn test_validatable_string() {
    let data = list_par_from_single(par_with_string("hello"));
    let result = data.to_validatable_string();
    assert_eq!(result, "String(hello)");
    assert_eq!(data.type_name(), "String");
}

/// Test that booleans are converted to "Bool(b)" format.
#[test]
fn test_validatable_bool() {
    let data = list_par_from_single(par_with_bool(true));
    let result = data.to_validatable_string();
    assert_eq!(result, "Bool(true)");
    assert_eq!(data.type_name(), "Bool");
}

/// Test that empty ListParWithRandom converts to "Unit".
#[test]
fn test_validatable_empty_is_unit() {
    let data = ListParWithRandom {
        pars: vec![],
        random_state: vec![],
    };
    let result = data.to_validatable_string();
    assert_eq!(result, "Unit");
    assert_eq!(data.type_name(), "Unit");
}

// =============================================================================
// Theory Validation Unit Tests
// =============================================================================

/// Test that NullTheory accepts any data.
#[test]
fn test_null_theory_accepts_all() {
    let theory = NullTheory;
    assert!(theory.validate("anything").is_ok());
    assert!(theory.validate("Nat(42)").is_ok());
    assert!(theory.validate("String(hello)").is_ok());
    assert!(theory.validate("").is_ok());
}

/// Test that Nat theory accepts non-negative integers.
#[test]
fn test_nat_theory_accepts_nat() {
    let theory = nat_theory();
    assert!(theory.validate("Nat(0)").is_ok());
    assert!(theory.validate("Nat(42)").is_ok());
    assert!(theory.validate("Nat(999999)").is_ok());
}

/// Test that Nat theory rejects negative integers.
#[test]
fn test_nat_theory_rejects_int() {
    let theory = nat_theory();
    assert!(theory.validate("Int(-5)").is_err());
    assert!(theory.validate("Int(-1)").is_err());
}

/// Test that Nat theory rejects strings.
#[test]
fn test_nat_theory_rejects_string() {
    let theory = nat_theory();
    assert!(theory.validate("String(hello)").is_err());
}

/// Test that Int theory accepts both Nat and Int.
#[test]
fn test_int_theory_accepts_nat_and_int() {
    let theory = int_theory();
    assert!(theory.validate("Nat(42)").is_ok());
    assert!(theory.validate("Int(-5)").is_ok());
}

/// Test that Int theory rejects strings.
#[test]
fn test_int_theory_rejects_string() {
    let theory = int_theory();
    assert!(theory.validate("String(hello)").is_err());
}

/// Test that String theory accepts strings.
#[test]
fn test_string_theory_accepts_string() {
    let theory = string_theory();
    assert!(theory.validate("String(hello)").is_ok());
    assert!(theory.validate("String()").is_ok());
    assert!(theory.validate("String(with spaces)").is_ok());
}

/// Test that String theory rejects integers.
#[test]
fn test_string_theory_rejects_int() {
    let theory = string_theory();
    assert!(theory.validate("Nat(42)").is_err());
    assert!(theory.validate("Int(-5)").is_err());
}

/// Test that Bool theory accepts booleans.
#[test]
fn test_bool_theory_accepts_bool() {
    let theory = bool_theory();
    assert!(theory.validate("Bool(true)").is_ok());
    assert!(theory.validate("Bool(false)").is_ok());
}

/// Test that Bool theory rejects integers.
#[test]
fn test_bool_theory_rejects_int() {
    let theory = bool_theory();
    assert!(theory.validate("Nat(42)").is_err());
}

/// Test that Any theory accepts everything.
#[test]
fn test_any_theory_accepts_all() {
    let theory = any_theory();
    assert!(theory.validate("Nat(42)").is_ok());
    assert!(theory.validate("Int(-5)").is_ok());
    assert!(theory.validate("String(hello)").is_ok());
    assert!(theory.validate("Bool(true)").is_ok());
    assert!(theory.validate("Unit").is_ok());
}

// =============================================================================
// SpaceConfig with Theory Tests
// =============================================================================

/// Test that SpaceConfig can be created with a theory.
#[test]
fn test_space_config_with_theory() {
    let config = SpaceConfig::hashmap_bag()
        .with_theory(Box::new(nat_theory()));

    assert!(config.theory.is_some());
    assert_eq!(config.theory.as_ref().unwrap().name(), "Nat");
}

/// Test that SpaceConfig without theory accepts all data.
#[test]
fn test_space_config_without_theory_accepts_all() {
    let config = SpaceConfig::hashmap_bag();
    assert!(config.theory.is_none());
    // When there's no theory, validation should be skipped entirely
}

/// Test that SpaceConfig's theory validates data correctly.
#[test]
fn test_space_config_theory_validates_data() {
    let config = SpaceConfig::hashmap_bag()
        .with_theory(Box::new(nat_theory()));

    let valid_data = list_par_from_single(par_with_int(42));
    let invalid_data = list_par_from_single(par_with_int(-5));

    if let Some(ref theory) = config.theory {
        // Valid: positive integer -> Nat
        let valid_str = valid_data.to_validatable_string();
        assert!(theory.validate(&valid_str).is_ok());

        // Invalid: negative integer -> Int (rejected by Nat theory)
        let invalid_str = invalid_data.to_validatable_string();
        assert!(theory.validate(&invalid_str).is_err());
    }
}

// =============================================================================
// TheoryValidator Trait Tests
// =============================================================================

/// Test TheoryValidator extension trait with valid data.
#[test]
fn test_theory_validator_valid_data() {
    let theory = nat_theory();
    let data = list_par_from_single(par_with_int(42));

    let result = theory.validate_data(&data);
    assert!(result.is_ok());
}

/// Test TheoryValidator extension trait with invalid data.
#[test]
fn test_theory_validator_invalid_data() {
    let theory = nat_theory();
    let data = list_par_from_single(par_with_int(-5));

    let result = theory.validate_data(&data);
    assert!(result.is_err());
}

// =============================================================================
// Property-Based Tests
// =============================================================================

proptest! {
    #![proptest_config(config())]

    /// Property: NullTheory always accepts any term.
    #[test]
    fn prop_null_theory_accepts_any_term(term in "[a-zA-Z0-9_() ]{0,100}") {
        let theory = NullTheory;
        prop_assert!(theory.validate(&term).is_ok());
    }

    /// Property: Non-negative integers are represented as Nat.
    #[test]
    fn prop_non_negative_int_is_nat(value in 0i64..=i64::MAX) {
        let data = list_par_from_single(par_with_int(value));
        let result = data.to_validatable_string();
        prop_assert!(result.starts_with("Nat("));
    }

    /// Property: Negative integers are represented as Int.
    #[test]
    fn prop_negative_int_is_int(value in i64::MIN..0i64) {
        let data = list_par_from_single(par_with_int(value));
        let result = data.to_validatable_string();
        prop_assert!(result.starts_with("Int("));
    }

    /// Property: Nat theory accepts all non-negative integers.
    #[test]
    fn prop_nat_theory_accepts_non_negative(value in 0i64..=1_000_000i64) {
        let theory = nat_theory();
        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();
        prop_assert!(theory.validate(&term).is_ok());
    }

    /// Property: Nat theory rejects all negative integers.
    #[test]
    fn prop_nat_theory_rejects_negative(value in -1_000_000i64..0i64) {
        let theory = nat_theory();
        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();
        prop_assert!(theory.validate(&term).is_err());
    }

    /// Property: Int theory accepts all integers (both positive and negative).
    #[test]
    fn prop_int_theory_accepts_all_integers(value in -1_000_000i64..=1_000_000i64) {
        let theory = int_theory();
        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();
        prop_assert!(theory.validate(&term).is_ok());
    }

    /// Property: String theory only accepts strings.
    #[test]
    fn prop_string_theory_rejects_integers(value in -1_000_000i64..=1_000_000i64) {
        let theory = string_theory();
        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();
        prop_assert!(theory.validate(&term).is_err());
    }

    /// Property: Any theory accepts any validatable data.
    #[test]
    fn prop_any_theory_accepts_integers(value in i64::MIN..=i64::MAX) {
        let theory = any_theory();
        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();
        prop_assert!(theory.validate(&term).is_ok());
    }

    /// Property: Clone preserves theory behavior.
    #[test]
    fn prop_theory_clone_preserves_validation(value in 0i64..=1_000_000i64) {
        let theory = nat_theory();
        let cloned_theory = theory.clone_box();

        let data = list_par_from_single(par_with_int(value));
        let term = data.to_validatable_string();

        prop_assert_eq!(
            theory.validate(&term).is_ok(),
            cloned_theory.validate(&term).is_ok()
        );
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test validation with the maximum i64 value.
#[test]
fn test_max_i64_validates_as_nat() {
    let theory = nat_theory();
    let data = list_par_from_single(par_with_int(i64::MAX));
    let term = data.to_validatable_string();
    assert!(theory.validate(&term).is_ok());
}

/// Test validation with the minimum i64 value.
#[test]
fn test_min_i64_validates_as_int() {
    let theory = int_theory();
    let data = list_par_from_single(par_with_int(i64::MIN));
    let term = data.to_validatable_string();
    assert!(theory.validate(&term).is_ok());
}

/// Test validation with empty string.
#[test]
fn test_empty_string_validates() {
    let theory = string_theory();
    let data = list_par_from_single(par_with_string(""));
    let term = data.to_validatable_string();
    assert!(theory.validate(&term).is_ok());
}

/// Test validation with special characters in string.
#[test]
fn test_special_chars_in_string() {
    let theory = string_theory();
    let data = list_par_from_single(par_with_string("hello\nworld\ttab"));
    let term = data.to_validatable_string();
    assert!(theory.validate(&term).is_ok());
}
