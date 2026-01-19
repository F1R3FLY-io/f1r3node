//! Space Identification
//!
//! This module defines the SpaceId type for uniquely identifying space instances.

use std::fmt;
use crypto::rust::hash::blake2b512_random::Blake2b512Random;

// ==========================================================================
// Space Identification
// ==========================================================================

/// Unique identifier for a space instance.
///
/// Space IDs are used to:
/// - Register spaces in the SpaceRegistry
/// - Route channel operations to the correct space
/// - Track channel ownership across spaces
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct SpaceId(pub Vec<u8>);

impl SpaceId {
    /// Create a new SpaceId from raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        SpaceId(bytes)
    }

    /// Generate a SpaceId from random state.
    /// Uses the same randomness source as gensym for correlation.
    pub fn from_random(rand: &mut Blake2b512Random) -> Self {
        // Use 32 bytes from the random generator
        let random_bytes = rand.next();
        SpaceId(random_bytes.iter().take(32).map(|&x| x as u8).collect())
    }

    /// The default/root space ID.
    /// This is the space that exists at runtime initialization.
    pub fn default_space() -> Self {
        SpaceId(vec![0u8; 32])
    }

    /// Convert to path bytes for PathMap indexing.
    pub fn to_path(&self) -> &[u8] {
        &self.0
    }

    /// Get the raw bytes of this SpaceId.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Create a SpaceId from a hex string.
    pub fn from_hex(hex_str: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(hex_str)?;
        Ok(SpaceId(bytes))
    }

    /// Convert to a hex string representation.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }
}

impl fmt::Display for SpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as truncated hex for readability
        let hex = self.to_hex();
        if hex.len() > 16 {
            write!(f, "SpaceId({}...)", &hex[..16])
        } else {
            write!(f, "SpaceId({})", hex)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_id_default() {
        let id = SpaceId::default_space();
        assert_eq!(id.0.len(), 32);
        assert!(id.0.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_space_id_hex_roundtrip() {
        let original = SpaceId::new(vec![0xde, 0xad, 0xbe, 0xef]);
        let hex = original.to_hex();
        let recovered = SpaceId::from_hex(&hex).expect("valid hex");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_space_id_display() {
        let short_id = SpaceId::new(vec![0xab, 0xcd]);
        assert_eq!(format!("{}", short_id), "SpaceId(abcd)");

        let long_id = SpaceId::new(vec![0xab; 20]);
        let display = format!("{}", long_id);
        assert!(display.contains("..."));
    }

    #[test]
    fn test_space_id_new() {
        let id = SpaceId::new(vec![1, 2, 3, 4]);
        assert_eq!(id.as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(id.to_path(), &[1, 2, 3, 4]);
    }
}
