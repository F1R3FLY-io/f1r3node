use crate::error::{NodeCliError, Result};
use crypto::rust::public_key::PublicKey;
use hex;
use rand::rngs::OsRng;
use rholang::rust::interpreter::util::rev_address::RevAddress;
use secp256k1::{PublicKey as Secp256k1PublicKey, Secp256k1, SecretKey};
use std::fs;
use std::path::Path;

pub struct CryptoUtils;

impl CryptoUtils {
    /// Decode a hex-encoded private key
    pub fn decode_private_key(private_key_hex: &str) -> Result<SecretKey> {
        let private_key_bytes = hex::decode(private_key_hex)?;

        SecretKey::from_slice(&private_key_bytes)
            .map_err(|e| NodeCliError::crypto_invalid_private_key(&e.to_string()))
    }

    /// Generate a new random key pair
    pub fn generate_key_pair() -> Result<(SecretKey, Secp256k1PublicKey)> {
        let secp = Secp256k1::new();
        let mut rng = OsRng::default();
        let secret_key = SecretKey::new(&mut rng);
        let public_key = secret_key.public_key(&secp);
        Ok((secret_key, public_key))
    }

    /// Derive public key from private key
    pub fn derive_public_key(private_key: &SecretKey) -> Secp256k1PublicKey {
        let secp = Secp256k1::new();
        private_key.public_key(&secp)
    }

    /// Serialize public key to hex (compressed or uncompressed)
    pub fn serialize_public_key(public_key: &Secp256k1PublicKey, compressed: bool) -> String {
        if compressed {
            hex::encode(public_key.serialize())
        } else {
            hex::encode(public_key.serialize_uncompressed())
        }
    }

    /// Serialize private key to hex
    pub fn serialize_private_key(private_key: &SecretKey) -> String {
        hex::encode(private_key.secret_bytes())
    }

    /// Generate REV address from public key
    pub fn generate_rev_address(public_key_hex: &str) -> Result<String> {
        let public_key_bytes = hex::decode(public_key_hex)?;

        let public_key = PublicKey::from_bytes(&public_key_bytes);

        match RevAddress::from_public_key(&public_key) {
            Some(rev_address) => Ok(rev_address.to_base58()),
            None => Err(NodeCliError::crypto_invalid_public_key(
                "Failed to generate REV address from public key",
            )),
        }
    }

    /// Write key pair to files
    pub fn write_key_pair_to_files(
        private_key: &SecretKey,
        public_key: &Secp256k1PublicKey,
        private_key_file: &Path,
        public_key_file: &Path,
        compressed: bool,
    ) -> Result<()> {
        let private_key_hex = Self::serialize_private_key(private_key);
        let public_key_hex = Self::serialize_public_key(public_key, compressed);

        fs::write(private_key_file, private_key_hex).map_err(|e| {
            NodeCliError::file_write_failed(&private_key_file.display().to_string(), &e.to_string())
        })?;

        fs::write(public_key_file, public_key_hex).map_err(|e| {
            NodeCliError::file_write_failed(&public_key_file.display().to_string(), &e.to_string())
        })?;

        Ok(())
    }

    /// Create secp256k1 context
    pub fn create_secp256k1_context() -> Secp256k1<secp256k1::All> {
        Secp256k1::new()
    }

    /// Validate hex string
    pub fn is_valid_hex(hex_str: &str) -> bool {
        hex::decode(hex_str).is_ok()
    }

    /// Validate private key format
    pub fn is_valid_private_key(private_key_hex: &str) -> bool {
        if let Ok(bytes) = hex::decode(private_key_hex) {
            SecretKey::from_slice(&bytes).is_ok()
        } else {
            false
        }
    }

    /// Validate public key format
    pub fn is_valid_public_key(public_key_hex: &str) -> bool {
        if let Ok(bytes) = hex::decode(public_key_hex) {
            // PublicKey::from_bytes always succeeds, so just check if hex decode worked
            // and if the bytes are a reasonable length for a public key
            bytes.len() == 65 || bytes.len() == 33 // uncompressed or compressed
        } else {
            false
        }
    }
}
