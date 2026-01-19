//! Error types for the multi-space RSpace integration.
//!
//! These errors cover space management, channel routing, and collection-specific
//! error conditions.

use std::fmt;
use super::types::SpaceId;

/// Errors that can occur during space operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaceError {
    /// The requested space was not found in the registry.
    SpaceNotFound {
        space_id: SpaceId,
    },

    /// The channel is not associated with any space.
    ChannelNotFound {
        description: String,
    },

    /// Attempted to use a channel in the wrong space.
    ChannelSpaceMismatch {
        channel_desc: String,
        expected_space: SpaceId,
        actual_space: SpaceId,
    },

    /// All channels in a join must be from the same space.
    CrossSpaceJoinNotAllowed {
        description: String,
    },

    /// A sequential (Seq) channel was used in a context that requires mobility.
    SeqChannelNotMobile {
        description: String,
    },

    /// Out of channel names in an Array-backed space.
    OutOfNames {
        space_id: SpaceId,
        max_size: usize,
    },

    /// Out of memory when allocating channels in a Vector-backed space.
    OutOfMemory {
        space_id: SpaceId,
        description: String,
    },

    /// Attempted to send to a Cell channel that already contains data.
    CellAlreadyFull {
        channel_desc: String,
    },

    /// The use block stack is empty (no default space set).
    NoDefaultSpace,

    /// Invalid space configuration.
    InvalidConfiguration {
        description: String,
    },

    /// Space factory error during creation.
    FactoryError {
        factory_name: String,
        description: String,
    },

    /// Builder was not fully configured before build() was called.
    ///
    /// This error occurs when a required field is missing from a builder.
    /// Use the builder's `with_*` methods to provide all required fields.
    BuilderIncomplete {
        /// Name of the builder that was incomplete
        builder: &'static str,
        /// Name of the missing required field
        missing_field: &'static str,
    },

    /// Checkpoint operation failed.
    CheckpointError {
        description: String,
    },

    /// Replay operation failed.
    ReplayError {
        description: String,
    },

    /// An operation was attempted on a space with incompatible qualifier.
    QualifierMismatch {
        expected: String,
        actual: String,
        operation: String,
    },

    /// Internal error (should not occur in normal operation).
    InternalError {
        description: String,
    },

    // ============================================================================
    // Phlogiston (Gas) Errors
    // Formal Correspondence: Phlogiston.v (phlogiston_non_negative, charge_preserves_non_negative)
    // ============================================================================

    /// Insufficient phlogiston (gas) to complete the operation.
    /// This is the primary resource exhaustion error.
    OutOfPhlogiston {
        /// Amount of phlogiston required for the operation
        required: u64,
        /// Amount of phlogiston available
        available: u64,
        /// Description of the operation that failed
        operation: String,
    },

    /// A phlogiston charge operation failed.
    PhlogistonChargeError {
        /// Amount that was attempted to be charged
        amount: u64,
        /// Current balance
        current_balance: u64,
        /// Description of why the charge failed
        reason: String,
    },

    /// Invalid phlogiston amount specified (e.g., negative or overflow).
    InvalidPhlogistonAmount {
        description: String,
    },

    /// Phlogiston accounting invariant violated.
    /// This indicates a bug in the phlogiston tracking system.
    PhlogistonInvariantViolation {
        description: String,
    },

    // ============================================================================
    // Theory/Grammar Errors
    // Formal Correspondence: Theory validation for typed tuple spaces
    // ============================================================================

    /// Data failed validation against the space's theory.
    ///
    /// This error occurs when a space has a theory attached and the data
    /// being produced does not conform to that theory.
    ///
    /// # Example
    /// A space with a "Nat" theory would reject negative numbers or strings.
    TheoryValidationError {
        /// The name of the theory that rejected the data
        theory_name: String,
        /// Description of why validation failed
        validation_error: String,
        /// Representation of the rejected term (for debugging)
        term: String,
    },

    /// The theory expected by the space does not match the actual theory.
    TheoryMismatch {
        /// The theory expected by the operation
        expected: String,
        /// The actual theory attached to the space
        actual: String,
    },

    /// An error occurred while parsing or loading a theory.
    TheoryParseError {
        /// Name or path of the theory being parsed
        theory_source: String,
        /// Description of the parse error
        error: String,
    },

    /// The theory is not supported or not available.
    TheoryNotSupported {
        /// Name of the unsupported theory
        theory_name: String,
        /// Reason why it's not supported
        reason: String,
    },

    // ============================================================================
    // Serialization Errors
    // Formal Correspondence: Checkpoint.v (checkpoint_preserves_state, replay_restores_state)
    // ============================================================================

    /// Deserialization of space state failed.
    ///
    /// This error occurs when loading a checkpoint or replaying from a log
    /// and the serialized data cannot be decoded.
    DeserializationError {
        /// Description of the deserialization error
        message: String,
    },

    /// Serialization of space state failed.
    ///
    /// This error occurs when creating a checkpoint and the space state
    /// cannot be encoded.
    SerializationError {
        /// Description of the serialization error
        message: String,
    },

    // ============================================================================
    // Similarity Matching Errors
    // Formal Correspondence: VectorDB.v (similarity_match_well_formed)
    // ============================================================================

    /// Error during VectorDB similarity matching.
    ///
    /// This error occurs when:
    /// - The similarity query embedding is malformed
    /// - The threshold is out of range
    /// - The embedding dimensions don't match the space
    /// - The space doesn't support similarity matching
    SimilarityMatchError {
        /// Description of the error
        reason: String,
    },

    /// Error extracting embedding from data during produce operation.
    ///
    /// This error occurs when:
    /// - The data is not a map with an "embedding" key
    /// - The embedding value is not in the expected format
    /// - The embedding contains invalid values for the configured type
    EmbeddingExtractionError {
        /// Description of what went wrong
        description: String,
    },

    /// Embedding dimension mismatch between data and space configuration.
    ///
    /// This error occurs when:
    /// - A produced embedding has different dimensions than the space expects
    /// - A query embedding has different dimensions than the space expects
    DimensionMismatch {
        /// Expected dimensions from space configuration
        expected: usize,
        /// Actual dimensions of the embedding
        actual: usize,
    },

    // ============================================================================
    // Function Handler Errors (Phase 8)
    // Formal Correspondence: Function handler registry for extensible VectorDB
    // ============================================================================

    /// Invalid argument passed to a function handler.
    ///
    /// This error occurs when:
    /// - A parameter has the wrong type (e.g., string instead of integer)
    /// - A required parameter is missing
    /// - A parameter value is out of range
    InvalidArgument(String),

    /// Unknown function identifier.
    ///
    /// This error occurs when:
    /// - A similarity metric ID is not registered
    /// - A ranking function ID is not registered
    UnknownFunction {
        /// The kind of function (similarity or ranking)
        kind: String,
        /// The unrecognized function identifier
        identifier: String,
    },

    /// Function arity mismatch.
    ///
    /// This error occurs when the wrong number of parameters are passed
    /// to a ranking function.
    ArityMismatch {
        /// Function identifier
        function: String,
        /// Expected (min, max) arity
        expected: (usize, usize),
        /// Actual number of parameters
        actual: usize,
    },
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpaceError::SpaceNotFound { space_id } => {
                write!(f, "Space not found: {}", space_id)
            }
            SpaceError::ChannelNotFound { description } => {
                write!(f, "Channel not found: {}", description)
            }
            SpaceError::ChannelSpaceMismatch {
                channel_desc,
                expected_space,
                actual_space,
            } => {
                write!(
                    f,
                    "Channel {} belongs to space {} but was used in space {}",
                    channel_desc, actual_space, expected_space
                )
            }
            SpaceError::CrossSpaceJoinNotAllowed { description } => {
                write!(
                    f,
                    "All channels in a join must be from the same space: {}",
                    description
                )
            }
            SpaceError::SeqChannelNotMobile { description } => {
                write!(
                    f,
                    "Sequential (seq) channels cannot be sent to other processes: {}",
                    description
                )
            }
            SpaceError::OutOfNames { space_id, max_size } => {
                write!(
                    f,
                    "Out of channel names in space {} (max size: {})",
                    space_id, max_size
                )
            }
            SpaceError::OutOfMemory { space_id, description } => {
                write!(
                    f,
                    "Out of memory in space {}: {}",
                    space_id, description
                )
            }
            SpaceError::CellAlreadyFull { channel_desc } => {
                write!(
                    f,
                    "Cell channel {} already contains data; cannot send again",
                    channel_desc
                )
            }
            SpaceError::NoDefaultSpace => {
                write!(f, "No default space set (use block stack is empty)")
            }
            SpaceError::InvalidConfiguration { description } => {
                write!(f, "Invalid space configuration: {}", description)
            }
            SpaceError::FactoryError {
                factory_name,
                description,
            } => {
                write!(f, "Space factory {} error: {}", factory_name, description)
            }
            SpaceError::BuilderIncomplete {
                builder,
                missing_field,
            } => {
                write!(
                    f,
                    "Builder '{}' incomplete: missing required field '{}'",
                    builder, missing_field
                )
            }
            SpaceError::CheckpointError { description } => {
                write!(f, "Checkpoint error: {}", description)
            }
            SpaceError::ReplayError { description } => {
                write!(f, "Replay error: {}", description)
            }
            SpaceError::QualifierMismatch {
                expected,
                actual,
                operation,
            } => {
                write!(
                    f,
                    "Qualifier mismatch for {}: expected {}, got {}",
                    operation, expected, actual
                )
            }
            SpaceError::InternalError { description } => {
                write!(f, "Internal space error: {}", description)
            }
            SpaceError::OutOfPhlogiston {
                required,
                available,
                operation,
            } => {
                write!(
                    f,
                    "Out of phlogiston: {} requires {} but only {} available",
                    operation, required, available
                )
            }
            SpaceError::PhlogistonChargeError {
                amount,
                current_balance,
                reason,
            } => {
                write!(
                    f,
                    "Phlogiston charge failed: cannot charge {} from balance of {}: {}",
                    amount, current_balance, reason
                )
            }
            SpaceError::InvalidPhlogistonAmount { description } => {
                write!(f, "Invalid phlogiston amount: {}", description)
            }
            SpaceError::PhlogistonInvariantViolation { description } => {
                write!(
                    f,
                    "Phlogiston invariant violated (bug): {}",
                    description
                )
            }
            SpaceError::TheoryValidationError {
                theory_name,
                validation_error,
                term,
            } => {
                write!(
                    f,
                    "Theory '{}' validation failed for term '{}': {}",
                    theory_name, term, validation_error
                )
            }
            SpaceError::TheoryMismatch { expected, actual } => {
                write!(
                    f,
                    "Theory mismatch: expected '{}', but space has '{}'",
                    expected, actual
                )
            }
            SpaceError::TheoryParseError { theory_source, error } => {
                write!(
                    f,
                    "Failed to parse theory '{}': {}",
                    theory_source, error
                )
            }
            SpaceError::TheoryNotSupported { theory_name, reason } => {
                write!(
                    f,
                    "Theory '{}' is not supported: {}",
                    theory_name, reason
                )
            }
            SpaceError::DeserializationError { message } => {
                write!(f, "Deserialization error: {}", message)
            }
            SpaceError::SerializationError { message } => {
                write!(f, "Serialization error: {}", message)
            }
            SpaceError::SimilarityMatchError { reason } => {
                write!(f, "Similarity matching error: {}", reason)
            }
            SpaceError::EmbeddingExtractionError { description } => {
                write!(f, "Embedding extraction error: {}", description)
            }
            SpaceError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Embedding dimension mismatch: expected {} dimensions, got {}",
                    expected, actual
                )
            }
            SpaceError::InvalidArgument(msg) => {
                write!(f, "Invalid argument: {}", msg)
            }
            SpaceError::UnknownFunction { kind, identifier } => {
                write!(f, "Unknown {} function: '{}'", kind, identifier)
            }
            SpaceError::ArityMismatch {
                function,
                expected,
                actual,
            } => {
                if expected.0 == expected.1 {
                    write!(
                        f,
                        "Function '{}' expects {} parameters, got {}",
                        function, expected.0, actual
                    )
                } else {
                    write!(
                        f,
                        "Function '{}' expects {}-{} parameters, got {}",
                        function, expected.0, expected.1, actual
                    )
                }
            }
        }
    }
}

impl std::error::Error for SpaceError {}

// ============================================================================
// Error Conversions
// ============================================================================

impl From<SpaceError> for super::super::errors::InterpreterError {
    fn from(err: SpaceError) -> Self {
        super::super::errors::InterpreterError::ReduceError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SpaceError::SpaceNotFound {
            space_id: SpaceId::default_space(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Space not found"));
    }

    #[test]
    fn test_out_of_names_display() {
        let err = SpaceError::OutOfNames {
            space_id: SpaceId::new(vec![1, 2, 3]),
            max_size: 100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Out of channel names"));
        assert!(msg.contains("100"));
    }
}
