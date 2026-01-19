//! Space Qualifier
//!
//! This module defines the SpaceQualifier enum that controls persistence
//! and concurrency behavior for spaces.

use std::fmt;

// ==========================================================================
// LAYER 3: Space Qualifier (persistence and concurrency)
// ==========================================================================

/// Qualifier for space behavior - controls persistence and concurrency.
///
/// This determines how channels in the space behave with respect to:
/// - Persistence: Whether data survives across checkpoints
/// - Concurrency: Whether parallel access is allowed
/// - Mobility: Whether channel references can be sent to other spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub enum SpaceQualifier {
    /// Persistent, concurrent access (default behavior).
    /// Data survives checkpoints, parallel processes can access simultaneously.
    #[default]
    Default,

    /// Non-persistent, concurrent access.
    /// Data is cleared on checkpoint, but parallel access is allowed.
    /// Useful for temporary computation that doesn't need to persist.
    Temp,

    /// Non-persistent, sequential, restricted.
    /// - Cannot be sent to other processes
    /// - No concurrent access allowed
    /// - Operations execute in strict sequence
    /// Used for local mutable state that must not escape.
    Seq,
}

impl SpaceQualifier {
    /// Check if channels with this qualifier can be sent to other processes.
    /// Seq channels are non-mobile and cannot be sent.
    pub fn is_mobile(&self) -> bool {
        !matches!(self, SpaceQualifier::Seq)
    }

    /// Check if this qualifier supports persistent storage.
    pub fn is_persistent(&self) -> bool {
        matches!(self, SpaceQualifier::Default)
    }

    /// Check if this qualifier supports concurrent access.
    pub fn is_concurrent(&self) -> bool {
        !matches!(self, SpaceQualifier::Seq)
    }
}

impl fmt::Display for SpaceQualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpaceQualifier::Default => write!(f, "default"),
            SpaceQualifier::Temp => write!(f, "temp"),
            SpaceQualifier::Seq => write!(f, "seq"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualifier_default() {
        let q = SpaceQualifier::Default;
        assert!(q.is_mobile());
        assert!(q.is_persistent());
        assert!(q.is_concurrent());
    }

    #[test]
    fn test_qualifier_temp() {
        let q = SpaceQualifier::Temp;
        assert!(q.is_mobile());
        assert!(!q.is_persistent());
        assert!(q.is_concurrent());
    }

    #[test]
    fn test_qualifier_seq() {
        let q = SpaceQualifier::Seq;
        assert!(!q.is_mobile());
        assert!(!q.is_persistent());
        assert!(!q.is_concurrent());
    }

    #[test]
    fn test_qualifier_display() {
        assert_eq!(format!("{}", SpaceQualifier::Default), "default");
        assert_eq!(format!("{}", SpaceQualifier::Temp), "temp");
        assert_eq!(format!("{}", SpaceQualifier::Seq), "seq");
    }
}
