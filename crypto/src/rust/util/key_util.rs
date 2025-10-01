use crate::rust::{private_key::PrivateKey, public_key::PublicKey};
use eyre::{Context, Result};
use hex;
use openssl::pkey::PKey;
use std::path::Path;

/// Key generation and file writing utilities
/// Equivalent to Scala's KeyUtil.writeKeys functionality
pub struct KeyUtil;

impl KeyUtil {
    /// Write encrypted private key, public key PEM, and public key hex files
    /// Equivalent to Scala's KeyUtil.writeKeys method
    pub fn write_keys<P: AsRef<Path>>(
        private_key: &PrivateKey,
        public_key: &PublicKey,
        password: &str,
        private_key_pem_path: P,
        public_key_pem_path: P,
        public_key_hex_path: P,
    ) -> Result<()> {
        // Create OpenSSL PKey from our private key bytes
        let pkey = Self::create_pkey_from_private_key(private_key)?;

        // Write encrypted private key PEM file
        Self::write_encrypted_private_key_pem(&pkey, password, private_key_pem_path)?;

        // Write public key PEM file
        Self::write_public_key_pem(&pkey, public_key_pem_path)?;

        // Write public key hex file
        Self::write_public_key_hex(public_key, public_key_hex_path)?;

        Ok(())
    }

    /// Create an OpenSSL PKey from our PrivateKey bytes
    fn create_pkey_from_private_key(
        private_key: &PrivateKey,
    ) -> Result<PKey<openssl::pkey::Private>> {
        // Convert our private key to PKCS#8 DER format
        use k256::ecdsa::SigningKey;
        use k256::elliptic_curve::generic_array::GenericArray;
        use pkcs8::EncodePrivateKey;

        let signing_key = SigningKey::from_bytes(&GenericArray::from_slice(&private_key.bytes))
            .map_err(|e| eyre::eyre!("Failed to create signing key: {}", e))?;

        let pkcs8_der = signing_key
            .to_pkcs8_der()
            .map_err(|e| eyre::eyre!("Failed to create PKCS#8 DER: {}", e))?;

        // Create OpenSSL PKey from PKCS#8 DER
        let pkey = PKey::private_key_from_der(pkcs8_der.as_bytes())
            .map_err(|e| eyre::eyre!("Failed to create PKey from DER: {}", e))?;

        Ok(pkey)
    }

    /// Write encrypted private key to PEM file
    /// Equivalent to: `openssl ec -in key.pem -out privateKey.pem -aes256`
    fn write_encrypted_private_key_pem<P: AsRef<Path>>(
        pkey: &PKey<openssl::pkey::Private>,
        password: &str,
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();

        // Export as encrypted PEM using AES-256-CBC (matching Scala's JcePEMEncryptorBuilder)
        let encrypted_pem = pkey
            .private_key_to_pem_pkcs8_passphrase(
                openssl::symm::Cipher::aes_256_cbc(),
                password.as_bytes(),
            )
            .map_err(|e| eyre::eyre!("Failed to create encrypted PEM: {}", e))?;

        // Write to file
        std::fs::write(path, encrypted_pem).with_context(|| {
            format!(
                "Failed to write encrypted private key to: {}",
                path.display()
            )
        })?;

        Ok(())
    }

    /// Write public key to PEM file
    /// Equivalent to: `openssl ec -in privateKey.pem -pubout >> publicKey.pem`
    fn write_public_key_pem<P: AsRef<Path>>(
        pkey: &PKey<openssl::pkey::Private>,
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();

        // Export public key as PEM
        let public_pem = pkey
            .public_key_to_pem()
            .map_err(|e| eyre::eyre!("Failed to create public key PEM: {}", e))?;

        // Write to file
        std::fs::write(path, public_pem)
            .with_context(|| format!("Failed to write public key PEM to: {}", path.display()))?;

        Ok(())
    }

    /// Write public key as hex string to file
    /// Equivalent to Scala's Base16.encode(pk.bytes) + "\n"
    fn write_public_key_hex<P: AsRef<Path>>(public_key: &PublicKey, path: P) -> Result<()> {
        let path = path.as_ref();

        // Convert public key bytes to hex (uppercase, matching Scala's Base16.encode)
        let hex_string = hex::encode_upper(&public_key.bytes) + "\n";

        // Write to file
        std::fs::write(path, hex_string)
            .with_context(|| format!("Failed to write public key hex to: {}", path.display()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::rust::signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg};

    use super::*;
    use tempfile::NamedTempFile;

    fn generate_and_write_keys<P: AsRef<Path>>(
        password: &str,
        private_key_pem_path: P,
        public_key_pem_path: P,
        public_key_hex_path: P,
    ) -> Result<(PrivateKey, PublicKey)> {
        // Generate new key pair
        let secp256k1 = Secp256k1;
        let (private_key, public_key) = secp256k1.new_key_pair();

        // Write all files
        KeyUtil::write_keys(
            &private_key,
            &public_key,
            password,
            private_key_pem_path,
            public_key_pem_path,
            public_key_hex_path,
        )?;

        Ok((private_key, public_key))
    }

    #[test]
    fn test_write_keys() {
        let password = "test_password_123";

        // Generate test key pair
        let secp256k1 = Secp256k1;
        let (private_key, public_key) = secp256k1.new_key_pair();

        // Create temporary files
        let private_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_hex_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write keys
        KeyUtil::write_keys(
            &private_key,
            &public_key,
            password,
            private_key_file.path(),
            public_key_file.path(),
            public_hex_file.path(),
        )
        .expect("Failed to write keys");

        // Verify files were created and contain expected content
        let private_content = std::fs::read_to_string(private_key_file.path())
            .expect("Failed to read private key file");
        assert!(private_content.contains("BEGIN ENCRYPTED PRIVATE KEY"));
        assert!(private_content.contains("END ENCRYPTED PRIVATE KEY"));

        let public_content = std::fs::read_to_string(public_key_file.path())
            .expect("Failed to read public key file");
        assert!(public_content.contains("BEGIN PUBLIC KEY"));
        assert!(public_content.contains("END PUBLIC KEY"));

        let hex_content =
            std::fs::read_to_string(public_hex_file.path()).expect("Failed to read hex file");
        let expected_hex = hex::encode_upper(&public_key.bytes) + "\n";
        assert_eq!(hex_content, expected_hex);
    }

    #[test]
    fn test_generate_and_write_keys() {
        let password = "test_password_456";

        // Create temporary files
        let private_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_hex_file = NamedTempFile::new().expect("Failed to create temp file");

        // Generate and write keys
        let (private_key, public_key) = generate_and_write_keys(
            password,
            private_key_file.path(),
            public_key_file.path(),
            public_hex_file.path(),
        )
        .expect("Failed to generate and write keys");

        // Verify we can read back the private key with the password
        let parsed_private_key = Secp256k1::parse_pem_file(private_key_file.path(), password)
            .expect("Failed to parse encrypted private key");

        assert_eq!(parsed_private_key.bytes, private_key.bytes);

        // Verify public key files
        let public_content = std::fs::read_to_string(public_key_file.path())
            .expect("Failed to read public key file");
        assert!(public_content.contains("BEGIN PUBLIC KEY"));

        let hex_content =
            std::fs::read_to_string(public_hex_file.path()).expect("Failed to read hex file");
        let expected_hex = hex::encode_upper(&public_key.bytes) + "\n";
        assert_eq!(hex_content, expected_hex);
    }

    #[test]
    fn test_wrong_password_fails() {
        let password = "correct_password";
        let wrong_password = "wrong_password";

        // Generate test key pair
        let secp256k1 = Secp256k1;
        let (private_key, public_key) = secp256k1.new_key_pair();

        // Create temporary files
        let private_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_key_file = NamedTempFile::new().expect("Failed to create temp file");
        let public_hex_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write keys with correct password
        KeyUtil::write_keys(
            &private_key,
            &public_key,
            password,
            private_key_file.path(),
            public_key_file.path(),
            public_hex_file.path(),
        )
        .expect("Failed to write keys");

        // Try to read with wrong password - should fail
        let result = Secp256k1::parse_pem_file(private_key_file.path(), wrong_password);
        assert!(result.is_err(), "Should fail with wrong password");
    }
}
