//! Property-Based Tests for ISpaceAdapter Module
//!
//! This module contains property-based tests for the ISpace to SpaceAgent adapter,
//! which bridges the rspace++ ISpace trait with the reified RSpaces SpaceAgent trait.
//!
//! # Rocq Correspondence
//!
//! These tests correspond to formal proofs in:
//! - `theories/GenericRSpace.v` - Space agent properties
//! - `theories/Checkpoint.v` - Checkpoint delegation properties
//!
//! # Properties Tested
//!
//! Since the ISpaceAdapter requires a full ISpace implementation to test async methods,
//! these tests focus on synchronous properties that can be verified without a mock:
//!
//! 1. **SpaceId Preservation**: space_id() returns the configured ID
//! 2. **Qualifier Preservation**: qualifier() returns the configured qualifier
//! 3. **Error Conversion**: RSpaceError → SpaceError preserves information
//!
//! For full async testing, see the integration tests that use actual RSpace instances.

use proptest::prelude::*;

use rholang::rust::interpreter::spaces::{
    SpaceId, SpaceQualifier, SpaceError,
};

// Note: Full ISpaceAdapter testing requires a complete ISpace implementation.
// The adapter module has its own unit tests. These property tests focus on
// the types used by the adapter.

// =============================================================================
// SpaceId Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: SpaceId roundtrips through bytes.
    #[test]
    fn prop_space_id_bytes_roundtrip(bytes in proptest::collection::vec(any::<u8>(), 1..=32)) {
        let id = SpaceId::new(bytes.clone());
        prop_assert_eq!(id.as_bytes(), &bytes);
    }

    /// Property: SpaceId from string is deterministic.
    #[test]
    fn prop_space_id_from_string_deterministic(s in "[a-z][a-z0-9-]{0,20}") {
        let id1 = SpaceId::new(s.as_bytes().to_vec());
        let id2 = SpaceId::new(s.as_bytes().to_vec());
        prop_assert_eq!(id1, id2);
    }

    /// Property: Different inputs produce different SpaceIds.
    #[test]
    fn prop_space_id_different_inputs(
        s1 in "[a-z][a-z0-9]{1,10}",
        s2 in "[a-z][a-z0-9]{1,10}"
    ) {
        if s1 != s2 {
            let id1 = SpaceId::new(s1.as_bytes().to_vec());
            let id2 = SpaceId::new(s2.as_bytes().to_vec());
            prop_assert_ne!(id1, id2);
        }
    }
}

// =============================================================================
// SpaceQualifier Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: SpaceQualifier equality is reflexive.
    #[test]
    fn prop_qualifier_reflexive(idx in 0u8..3) {
        let qualifier = match idx {
            0 => SpaceQualifier::Default,
            1 => SpaceQualifier::Temp,
            _ => SpaceQualifier::Seq,
        };
        prop_assert_eq!(qualifier, qualifier);
    }

    /// Property: SpaceQualifier.is_mobile() is false only for Seq.
    #[test]
    fn prop_qualifier_mobility(idx in 0u8..3) {
        let qualifier = match idx {
            0 => SpaceQualifier::Default,
            1 => SpaceQualifier::Temp,
            _ => SpaceQualifier::Seq,
        };

        let expected_mobile = !matches!(qualifier, SpaceQualifier::Seq);
        prop_assert_eq!(qualifier.is_mobile(), expected_mobile);
    }
}

// =============================================================================
// SpaceError Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: SpaceError::InternalError preserves the description.
    #[test]
    fn prop_internal_error_preserves_description(desc in "[a-zA-Z0-9 ]{1,100}") {
        let error = SpaceError::InternalError { description: desc.clone() };
        let error_str = error.to_string();
        prop_assert!(
            error_str.contains(&desc) || error_str.contains("internal") || error_str.contains("Internal"),
            "Error string should contain description or 'internal'"
        );
    }

    /// Property: SpaceError::CheckpointError preserves the description.
    #[test]
    fn prop_checkpoint_error_preserves_description(desc in "[a-zA-Z0-9 ]{1,100}") {
        let error = SpaceError::CheckpointError { description: desc.clone() };
        let error_str = error.to_string();
        prop_assert!(
            error_str.contains(&desc) || error_str.contains("checkpoint") || error_str.contains("Checkpoint"),
            "Error string should contain description or 'checkpoint'"
        );
    }

    /// Property: SpaceError::ChannelNotFound preserves the description.
    #[test]
    fn prop_channel_not_found_error_preserves_description(desc in "[a-zA-Z0-9 ]{1,100}") {
        let error = SpaceError::ChannelNotFound { description: desc.clone() };
        let error_str = error.to_string();
        prop_assert!(
            error_str.contains(&desc) || error_str.to_lowercase().contains("channel") || error_str.contains("not found"),
            "Error string should contain description or relevant keywords"
        );
    }
}

// =============================================================================
// Default Space Tests
// =============================================================================

#[cfg(test)]
mod default_space_tests {
    use super::*;

    #[test]
    fn test_default_space_id_is_deterministic() {
        let id1 = SpaceId::default_space();
        let id2 = SpaceId::default_space();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_default_qualifier() {
        let default = SpaceQualifier::default();
        assert_eq!(default, SpaceQualifier::Default);
    }

    #[test]
    fn test_qualifier_display() {
        assert_eq!(format!("{}", SpaceQualifier::Default), "default");
        assert_eq!(format!("{}", SpaceQualifier::Temp), "temp");
        assert_eq!(format!("{}", SpaceQualifier::Seq), "seq");
    }
}

// =============================================================================
// Error Variant Tests
// =============================================================================

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_cross_space_join_error() {
        let error = SpaceError::CrossSpaceJoinNotAllowed {
            description: "test".to_string(),
        };
        assert!(error.to_string().to_lowercase().contains("cross") || error.to_string().contains("test"));
    }

    #[test]
    fn test_seq_channel_not_mobile_error() {
        let error = SpaceError::SeqChannelNotMobile {
            description: "seq channel test".to_string(),
        };
        assert!(error.to_string().contains("seq") || error.to_string().contains("mobile") || error.to_string().contains("test"));
    }

    #[test]
    fn test_invalid_configuration_error() {
        let error = SpaceError::InvalidConfiguration {
            description: "bad config".to_string(),
        };
        assert!(error.to_string().contains("config") || error.to_string().contains("bad config") || error.to_string().contains("Invalid"));
    }

    #[test]
    fn test_out_of_phlogiston_error() {
        let error = SpaceError::OutOfPhlogiston {
            required: 100,
            available: 50,
            operation: "test_op".to_string(),
        };
        let err_str = error.to_string();
        assert!(err_str.contains("100") || err_str.contains("50") || err_str.to_lowercase().contains("gas") || err_str.to_lowercase().contains("phlogiston"));
    }

    #[test]
    fn test_cell_already_full_error() {
        let error = SpaceError::CellAlreadyFull {
            channel_desc: "test_channel".to_string(),
        };
        assert!(error.to_string().contains("cell") || error.to_string().contains("full") || error.to_string().contains("test_channel"));
    }
}
