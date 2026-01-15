//! Theory and Validatable Traits for MeTTaIL Integration
//!
//! This module defines the fundamental types for type/contract validation of data.
//! When integrated with MeTTaIL, this enables typed tuple spaces where only
//! well-typed data can be stored.

use std::fmt;

use models::rhoapi::{expr::ExprInstance, EList, Expr, ListParWithRandom, Par};

// ==========================================================================
// Theory Trait (for MeTTaIL integration)
// ==========================================================================

/// Boxed theory for type erasure in SpaceConfig.
pub type BoxedTheory = Box<dyn Theory>;

/// Theory trait for type/contract validation of data.
///
/// This trait allows spaces to validate data against a type theory or contract
/// before accepting it. In the full implementation, this integrates with
/// MeTTaIL for rich type theory support including dependent types and contracts.
///
/// # Spec Reference
/// From the Reifying RSpaces spec: "Each space can optionally be associated with
/// a MeTTaIL theory that validates data before it enters the space."
///
/// # Example
/// ```ignore
/// // Future integration with MeTTaIL
/// use mettail::Theory as MettailTheory;
///
/// let theory = MettailTheory::parse("(: Nat Type)")?;
/// let config = SpaceConfig::default().with_theory(Box::new(theory));
/// ```
pub trait Theory: Send + Sync + fmt::Debug {
    /// Validate that the given term conforms to this theory.
    ///
    /// Returns `Ok(())` if the term is valid according to the theory,
    /// or `Err(description)` if validation fails.
    ///
    /// # Arguments
    /// * `term` - The term to validate, typically serialized as a string
    fn validate(&self, term: &str) -> Result<(), String>;

    /// Get the name/identifier of this theory.
    ///
    /// Used for debugging and error messages.
    fn name(&self) -> &str;

    /// Check if a type is defined in this theory.
    ///
    /// # Arguments
    /// * `type_name` - The name of the type to check
    fn has_type(&self, type_name: &str) -> bool {
        // Default implementation - override for actual type checking
        let _ = type_name;
        false
    }

    /// Get a human-readable description of the theory.
    fn description(&self) -> &str {
        "Unnamed theory"
    }

    /// Clone this theory into a boxed trait object.
    ///
    /// This is needed because trait objects cannot implement Clone directly.
    fn clone_box(&self) -> BoxedTheory;
}

/// A null theory that accepts all data (no validation).
///
/// This is the default when no theory is specified.
#[derive(Clone, Debug, Default)]
pub struct NullTheory;

impl Theory for NullTheory {
    fn validate(&self, _term: &str) -> Result<(), String> {
        Ok(()) // Accept everything
    }

    fn name(&self) -> &str {
        "NullTheory"
    }

    fn description(&self) -> &str {
        "Accepts all data without validation"
    }

    fn clone_box(&self) -> BoxedTheory {
        Box::new(self.clone())
    }
}

// ==========================================================================
// Validatable Trait
// ==========================================================================

/// Trait for data types that can be validated against a theory.
///
/// This trait allows data to be validated before being stored in a typed
/// tuple space. Types implementing this trait can be serialized to a string
/// representation that the theory can validate.
///
/// # Example
/// ```ignore
/// impl Validatable for MyData {
///     fn to_validatable_string(&self) -> String {
///         format!("MyData({})", self.value)
///     }
///
///     fn type_name(&self) -> &str {
///         "MyData"
///     }
/// }
/// ```
pub trait Validatable {
    /// Convert the data to a string representation for theory validation.
    ///
    /// This string is passed to `Theory::validate()` to check conformance.
    fn to_validatable_string(&self) -> String;

    /// Get the type name for this data.
    ///
    /// Used for error messages and theory type checking.
    fn type_name(&self) -> &str;

    /// Validate this data against a theory.
    ///
    /// Returns `Ok(())` if validation passes, or an error message if it fails.
    fn validate(&self, theory: &dyn Theory) -> Result<(), String> {
        theory.validate(&self.to_validatable_string())
    }
}

/// A validation result containing either success or a detailed error.
pub type ValidationResult = Result<(), super::super::errors::SpaceError>;

/// Extension trait for validating data before insertion into a typed space.
pub trait TheoryValidator {
    /// Validate data against this theory, returning a SpaceError on failure.
    fn validate_data<V: Validatable>(&self, data: &V) -> ValidationResult;
}

impl<T: Theory + ?Sized> TheoryValidator for T {
    fn validate_data<V: Validatable>(&self, data: &V) -> ValidationResult {
        match self.validate(&data.to_validatable_string()) {
            Ok(()) => Ok(()),
            Err(validation_error) => Err(super::super::errors::SpaceError::TheoryValidationError {
                theory_name: self.name().to_string(),
                validation_error,
                term: data.to_validatable_string(),
            }),
        }
    }
}

// ==========================================================================
// Theory Implementations
// ==========================================================================

/// A simple type theory that validates against a list of allowed type names.
///
/// This is a placeholder for testing until full MeTTaIL integration.
#[derive(Clone, Debug)]
pub struct SimpleTypeTheory {
    name: String,
    allowed_types: Vec<String>,
}

impl SimpleTypeTheory {
    /// Create a new SimpleTypeTheory with the given name and allowed types.
    pub fn new(name: impl Into<String>, allowed_types: Vec<String>) -> Self {
        SimpleTypeTheory {
            name: name.into(),
            allowed_types,
        }
    }
}

impl Theory for SimpleTypeTheory {
    fn validate(&self, term: &str) -> Result<(), String> {
        // Simple validation: check if term starts with an allowed type
        for allowed in &self.allowed_types {
            if term.starts_with(allowed) {
                return Ok(());
            }
        }
        Err(format!(
            "Term '{}' does not match any allowed type: {:?}",
            term, self.allowed_types
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn has_type(&self, type_name: &str) -> bool {
        self.allowed_types.iter().any(|t| t == type_name)
    }

    fn description(&self) -> &str {
        "Simple type validation against allowed type names"
    }

    fn clone_box(&self) -> BoxedTheory {
        Box::new(self.clone())
    }
}

// ==========================================================================
// Validatable Implementation for ListParWithRandom
// ==========================================================================

impl Validatable for ListParWithRandom {
    fn to_validatable_string(&self) -> String {
        // Convert the ListParWithRandom to a type-prefixed string for validation.
        // This examines the actual data and produces strings like:
        // - "Nat(42)" for non-negative integers
        // - "Int(-5)" for negative integers
        // - "String(hello)" for strings
        // - "Bool(true)" for booleans
        // - "Unknown" for complex/unknown data
        //
        // For multiple pars, we concatenate their representations.

        if self.pars.is_empty() {
            return "Unit".to_string();
        }

        let mut parts = Vec::new();
        for par in &self.pars {
            parts.push(par_to_validatable_type(par));
        }

        if parts.len() == 1 {
            parts.into_iter().next().expect("expected at least one part")
        } else {
            format!("Tuple({})", parts.join(", "))
        }
    }

    fn type_name(&self) -> &str {
        // Determine the primary type name based on contents
        if self.pars.is_empty() {
            return "Unit";
        }
        if self.pars.len() > 1 {
            return "Tuple";
        }

        // Single par - determine its type
        par_type_name(&self.pars[0])
    }
}

/// Convert a Par to a type-prefixed validation string.
fn par_to_validatable_type(par: &Par) -> String {
    // Check expressions first (most common case for data)
    for expr in &par.exprs {
        if let Some(ref instance) = expr.expr_instance {
            return expr_instance_to_validatable(instance);
        }
    }

    // Check other Par fields
    if !par.sends.is_empty() {
        return "Process(Send)".to_string();
    }
    if !par.receives.is_empty() {
        return "Process(Receive)".to_string();
    }
    if !par.news.is_empty() {
        return "Process(New)".to_string();
    }
    if !par.matches.is_empty() {
        return "Process(Match)".to_string();
    }
    if !par.unforgeables.is_empty() {
        return "Unforgeable".to_string();
    }
    if !par.bundles.is_empty() {
        return "Bundle".to_string();
    }
    if !par.connectives.is_empty() {
        return "Connective".to_string();
    }

    // Empty or unknown
    "Unknown".to_string()
}

/// Convert an ExprInstance to a type-prefixed validation string.
fn expr_instance_to_validatable(instance: &ExprInstance) -> String {
    match instance {
        ExprInstance::GInt(n) => {
            if *n >= 0 {
                format!("Nat({})", n)
            } else {
                format!("Int({})", n)
            }
        }
        ExprInstance::GString(s) => format!("String({})", s),
        ExprInstance::GBool(b) => format!("Bool({})", b),
        ExprInstance::GUri(u) => format!("Uri({})", u),
        ExprInstance::GByteArray(bytes) => format!("ByteArray(len={})", bytes.len()),

        // Collections
        ExprInstance::EListBody(elist) => {
            let items: Vec<String> = elist.ps.iter().map(par_to_validatable_type).collect();
            format!("List({})", items.join(", "))
        }
        ExprInstance::ETupleBody(etuple) => {
            let items: Vec<String> = etuple.ps.iter().map(par_to_validatable_type).collect();
            format!("Tuple({})", items.join(", "))
        }
        ExprInstance::ESetBody(eset) => {
            let items: Vec<String> = eset.ps.iter().map(par_to_validatable_type).collect();
            format!("Set({})", items.join(", "))
        }
        ExprInstance::EMapBody(emap) => {
            format!("Map(len={})", emap.kvs.len())
        }

        // Variables
        ExprInstance::EVarBody(_) => "Var".to_string(),
        ExprInstance::EFreeBody(_) => "Free".to_string(),

        // Arithmetic operations
        ExprInstance::EPlusBody(_)
        | ExprInstance::EMinusBody(_)
        | ExprInstance::EMultBody(_)
        | ExprInstance::EDivBody(_)
        | ExprInstance::EModBody(_)
        | ExprInstance::ENegBody(_) => "ArithExpr".to_string(),

        // Comparison operations
        ExprInstance::ELtBody(_)
        | ExprInstance::ELteBody(_)
        | ExprInstance::EGtBody(_)
        | ExprInstance::EGteBody(_)
        | ExprInstance::EEqBody(_)
        | ExprInstance::ENeqBody(_) => "CompareExpr".to_string(),

        // Logical operations
        ExprInstance::EAndBody(_)
        | ExprInstance::EOrBody(_)
        | ExprInstance::ENotBody(_) => "LogicalExpr".to_string(),

        // Collection operations
        ExprInstance::EPlusPlusBody(_) | ExprInstance::EMinusMinusBody(_) => {
            "CollectionExpr".to_string()
        }

        // String operations
        ExprInstance::EPercentPercentBody(_) => "StringExpr".to_string(),

        // Method call
        ExprInstance::EMethodBody(_) => "MethodCall".to_string(),

        // Function call
        ExprInstance::EFunctionBody(_) => "FunctionCall".to_string(),

        // Match expression
        ExprInstance::EMatchesBody(_) => "MatchExpr".to_string(),

        // PathMap
        ExprInstance::EPathmapBody(_) => "PathMap".to_string(),

        // Zipper
        ExprInstance::EZipperBody(_) => "Zipper".to_string(),
    }
}

/// Get the primary type name for a Par.
fn par_type_name(par: &Par) -> &'static str {
    for expr in &par.exprs {
        if let Some(ref instance) = expr.expr_instance {
            return match instance {
                ExprInstance::GInt(n) => {
                    if *n >= 0 {
                        "Nat"
                    } else {
                        "Int"
                    }
                }
                ExprInstance::GString(_) => "String",
                ExprInstance::GBool(_) => "Bool",
                ExprInstance::GUri(_) => "Uri",
                ExprInstance::GByteArray(_) => "ByteArray",
                ExprInstance::EListBody(_) => "List",
                ExprInstance::ETupleBody(_) => "Tuple",
                ExprInstance::ESetBody(_) => "Set",
                ExprInstance::EMapBody(_) => "Map",
                _ => "Expr",
            };
        }
    }
    "Unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_theory_accepts_everything() {
        let theory = NullTheory;
        assert!(theory.validate("anything").is_ok());
        assert!(theory.validate("").is_ok());
        assert!(theory.validate("(+ 1 2)").is_ok());
        assert_eq!(theory.name(), "NullTheory");
    }

    #[test]
    fn test_simple_type_theory_validates() {
        let theory = SimpleTypeTheory::new(
            "NatOrBool",
            vec!["Nat".to_string(), "Bool".to_string()],
        );

        // Valid terms
        assert!(theory.validate("Nat 42").is_ok());
        assert!(theory.validate("Bool true").is_ok());

        // Invalid terms
        assert!(theory.validate("String hello").is_err());
        assert!(theory.validate("Float 3.14").is_err());

        // Type checks
        assert!(theory.has_type("Nat"));
        assert!(theory.has_type("Bool"));
        assert!(!theory.has_type("String"));

        assert_eq!(theory.name(), "NatOrBool");
    }
}
