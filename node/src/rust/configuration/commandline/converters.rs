// TODO: double check whether this converters will be needed in future because it seems like that configs values where they are used are not in use

//! Value converters for command-line arguments
//!
//! This module provides custom converters for parsing command-line arguments
//! into specific types, similar to the Scala Base16Converter.

use eyre::Context;
use hex;

/// Base16 (hexadecimal) converter for byte arrays
pub struct Base16Converter;

impl Base16Converter {
    /// Convert a hex string to a byte array
    pub fn from_hex(hex_str: &str) -> eyre::Result<Vec<u8>> {
        hex::decode(hex_str).wrap_err("Invalid base16 encoding")
    }

    /// Convert a byte array to a hex string
    pub fn to_hex(bytes: &[u8]) -> String {
        hex::encode(bytes)
    }

    /// Validate that a hex string has the expected length when decoded
    pub fn validate_length(hex_str: &str, expected_length: usize) -> eyre::Result<Vec<u8>> {
        let bytes = Self::from_hex(hex_str)?;
        if bytes.len() == expected_length {
            Ok(bytes)
        } else {
            eyre::bail!(
                "Invalid parameter length. Expected length is {} bytes, got {} bytes",
                expected_length,
                bytes.len()
            );
        }
    }
}

/// Name type converter for Rholang names
pub struct NameConverter;

impl NameConverter {
    /// Convert string to public name
    pub fn to_pub_name(name: &str) -> crate::rust::configuration::model::Name {
        crate::rust::configuration::model::Name::PubName(name.to_string())
    }

    /// Convert string to private name
    pub fn to_priv_name(name: &str) -> crate::rust::configuration::model::Name {
        crate::rust::configuration::model::Name::PrivName(name.to_string())
    }

    /// Parse name with type specification
    pub fn parse_with_type(
        name_type: &str,
        content: &str,
    ) -> eyre::Result<crate::rust::configuration::model::Name> {
        match name_type {
            "pub" => Ok(Self::to_pub_name(content)),
            "priv" => Ok(Self::to_priv_name(content)),
            _ => eyre::bail!("Bad option value. Use \"pub\" or \"priv\""),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base16_converter() {
        let hex_str = "deadbeef";
        let bytes = Base16Converter::from_hex(hex_str).unwrap();
        assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef]);

        let back_to_hex = Base16Converter::to_hex(&bytes);
        assert_eq!(back_to_hex, hex_str);
    }

    #[test]
    fn test_name_converter() {
        let pub_name = NameConverter::to_pub_name("test");
        match pub_name {
            crate::rust::configuration::model::Name::PubName(name) => assert_eq!(name, "test"),
            _ => panic!("Expected PubName"),
        }

        let priv_name = NameConverter::to_priv_name("test");
        match priv_name {
            crate::rust::configuration::model::Name::PrivName(name) => assert_eq!(name, "test"),
            _ => panic!("Expected PrivName"),
        }
    }
}
