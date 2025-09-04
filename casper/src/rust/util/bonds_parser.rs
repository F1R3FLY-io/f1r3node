use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crypto::rust::{
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use models::rust::string_ops::StringOps;
use regex::Regex;

#[derive(Debug, thiserror::Error)]
pub enum BondsParserError {
    #[error("INVALID LINE FORMAT: `<public_key> <stake>`, actual: `{line}`")]
    InvalidLineFormat { line: String },
    #[error("INVALID PUBLIC KEY: `{key}`")]
    InvalidPublicKey { key: String },
    #[error("INVALID STAKE `{stake}`. Please put positive number.")]
    InvalidStake { stake: String },
    #[error("FAILED PARSING BONDS FILE: {path}\n{source}")]
    ParsingFailed {
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
}

pub struct BondsParser;

impl BondsParser {
    /// Parser for bonds file used in genesis ceremony to set initial validators.
    ///
    /// TODO: Create async file operations. For now it's ok because it's used only once at genesis.
    pub fn parse(bonds_path: &Path) -> Result<HashMap<PublicKey, i64>, BondsParserError> {
        log::info!("Parsing bonds file {:?}.", bonds_path);

        let content =
            fs::read_to_string(bonds_path).map_err(|e| BondsParserError::ParsingFailed {
                path: bonds_path.to_string_lossy().to_string(),
                source: Box::new(e),
            })?;

        let line_regex = Regex::new(r"^([0-9a-fA-F]+) ([0-9]+)").unwrap();
        let mut bonds = HashMap::new();

        for line in content.lines() {
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            // Parse line format
            let captures = line_regex.captures(trimmed_line).ok_or_else(|| {
                BondsParserError::InvalidLineFormat {
                    line: trimmed_line.to_string(),
                }
            })?;

            let public_key_str = captures
                .get(1)
                .ok_or_else(|| BondsParserError::InvalidLineFormat {
                    line: trimmed_line.to_string(),
                })?
                .as_str();

            let stake_str = captures
                .get(2)
                .ok_or_else(|| BondsParserError::InvalidLineFormat {
                    line: trimmed_line.to_string(),
                })?
                .as_str();

            // Parse public key
            let public_key_bytes =
                StringOps::decode_hex(public_key_str.to_string()).ok_or_else(|| {
                    BondsParserError::InvalidPublicKey {
                        key: public_key_str.to_string(),
                    }
                })?;
            let public_key = PublicKey::from_bytes(&public_key_bytes);

            // Parse stake
            let stake = stake_str
                .parse::<i64>()
                .map_err(|_| BondsParserError::InvalidStake {
                    stake: stake_str.to_string(),
                })?;

            log::info!(
                "Bond loaded {} => {}",
                hex::encode(&public_key.bytes),
                stake
            );

            bonds.insert(public_key, stake);
        }

        Ok(bonds)
    }

    pub fn parse_with_autogen(
        bonds_path_str: &str,
        autogen_shard_size: usize,
    ) -> Result<HashMap<PublicKey, i64>, BondsParserError> {
        let bonds_path = Path::new(bonds_path_str);

        if bonds_path.exists() {
            log::info!("Parsing bonds file {:?}.", bonds_path);
            Self::parse(bonds_path)
        } else {
            log::warn!(
                "BONDS FILE NOT FOUND: {:?}. Creating file with random bonds.",
                bonds_path
            );
            Self::new_validators(autogen_shard_size, bonds_path)
        }
    }

    fn new_validators(
        autogen_shard_size: usize,
        bonds_file_path: &Path,
    ) -> Result<HashMap<PublicKey, i64>, BondsParserError> {
        let genesis_folder = bonds_file_path.parent().ok_or_else(|| {
            BondsParserError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "bonds file path has no parent directory",
            ))
        })?;

        // Generate private/public key pairs
        let secp256k1 = Secp256k1;
        let keys: Vec<(PrivateKey, PublicKey)> = (0..autogen_shard_size)
            .map(|_| secp256k1.new_key_pair())
            .collect();

        let mut bonds = HashMap::new();
        for (i, (_, public_key)) in keys.iter().enumerate() {
            bonds.insert(public_key.clone(), (i as i64) + 1);
        }

        // Create genesis directory if it doesn't exist
        fs::create_dir_all(genesis_folder)?;

        // Write generated `<public_key>.sk` files with private key as content
        for (private_key, public_key) in &keys {
            let sk = hex::encode(&private_key.bytes);
            let pk = hex::encode(&public_key.bytes);
            let sk_file = genesis_folder.join(format!("{}.sk", pk));
            fs::write(&sk_file, &sk)?;
        }

        // Create bonds file with generated public keys
        let mut bonds_content = String::new();
        for (public_key, stake) in &bonds {
            let pk = hex::encode(&public_key.bytes);
            log::info!("Bond generated {} => {}", pk, stake);
            bonds_content.push_str(&format!("{} {}\n", pk, stake));
        }

        fs::write(bonds_file_path, bonds_content)?;

        Ok(bonds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_valid_bonds_file() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let bonds_content = "04f026cc75502aa420f589ba1437d52eb9055ba26701f112ff1b9a0b98cadec54bc3e77bdef9094b33f6afacd59eb1aed78bc1e1320ce714ec9debaabaf0f37ba3 150500000000000\n04ce255756a60b7847e3c8dd6383d8bed9b83933350fb380525e328d9f1e3152e7371ce6f0d02e1cc1fe9ea07877e05a4c42e1b3108721f261cf22ef51d3503d53 150500000000000\n";
        fs::write(&bonds_file, bonds_content).unwrap();

        let result = BondsParser::parse(&bonds_file);
        assert!(result.is_ok());

        let bonds = result.unwrap();
        assert_eq!(bonds.len(), 2);

        for (_, stake) in bonds {
            assert!(stake > 0);
        }
    }

    #[test]
    fn test_parse_invalid_line_format() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let invalid_content = "xzy 1\nabc 123 7\n";
        fs::write(&bonds_file, invalid_content).unwrap();

        let result = BondsParser::parse(&bonds_file);
        assert!(result.is_err());

        match result.unwrap_err() {
            BondsParserError::InvalidLineFormat { line } => {
                assert_eq!(line, "xzy 1");
            }
            _ => panic!("Expected InvalidLineFormat error"),
        }
    }

    #[test]
    fn test_parse_invalid_public_key() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let invalid_content = "invalidhexkey 1000\n";
        fs::write(&bonds_file, invalid_content).unwrap();

        let result = BondsParser::parse(&bonds_file);
        assert!(result.is_err());

        match result.unwrap_err() {
            BondsParserError::InvalidLineFormat { line } => {
                assert_eq!(line, "invalidhexkey 1000");
            }
            _ => panic!("Expected InvalidLineFormat error"),
        }
    }

    #[test]
    fn test_parse_invalid_stake() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let invalid_content = "04f026cc75502aa420f589ba1437d52eb9055ba26701f112ff1b9a0b98cadec54bc3e77bdef9094b33f6afacd59eb1aed78bc1e1320ce714ec9debaabaf0f37ba3 notanumber\n";
        fs::write(&bonds_file, invalid_content).unwrap();

        let result = BondsParser::parse(&bonds_file);
        assert!(result.is_err());

        match result.unwrap_err() {
            BondsParserError::InvalidLineFormat { line } => {
                assert_eq!(line, "04f026cc75502aa420f589ba1437d52eb9055ba26701f112ff1b9a0b98cadec54bc3e77bdef9094b33f6afacd59eb1aed78bc1e1320ce714ec9debaabaf0f37ba3 notanumber");
            }
            _ => panic!("Expected InvalidLineFormat error"),
        }
    }

    #[test]
    fn test_parse_with_autogen_existing_file() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let bonds_content = "04f026cc75502aa420f589ba1437d52eb9055ba26701f112ff1b9a0b98cadec54bc3e77bdef9094b33f6afacd59eb1aed78bc1e1320ce714ec9debaabaf0f37ba3 150500000000000\n";
        fs::write(&bonds_file, bonds_content).unwrap();

        let result = BondsParser::parse_with_autogen(bonds_file.to_str().unwrap(), 5);
        assert!(result.is_ok());

        let bonds = result.unwrap();
        assert_eq!(bonds.len(), 1);
    }

    #[test]
    fn test_parse_with_autogen_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("nonexistent_bonds.txt");

        let result = BondsParser::parse_with_autogen(bonds_file.to_str().unwrap(), 3);
        assert!(result.is_ok());

        let bonds = result.unwrap();
        assert_eq!(bonds.len(), 3);

        // Check that bonds file was created
        assert!(bonds_file.exists());

        // Check that .sk files were created
        for (public_key, _) in bonds {
            let pk_hex = hex::encode(&public_key.bytes);
            let sk_file = temp_dir.path().join(format!("{}.sk", pk_hex));
            assert!(sk_file.exists());
        }
    }

    #[test]
    fn test_empty_lines_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let bonds_file = temp_dir.path().join("bonds.txt");

        let bonds_content = "\n04f026cc75502aa420f589ba1437d52eb9055ba26701f112ff1b9a0b98cadec54bc3e77bdef9094b33f6afacd59eb1aed78bc1e1320ce714ec9debaabaf0f37ba3 150500000000000\n\n   \n04ce255756a60b7847e3c8dd6383d8bed9b83933350fb380525e328d9f1e3152e7371ce6f0d02e1cc1fe9ea07877e05a4c42e1b3108721f261cf22ef51d3503d53 150500000000000\n\n";
        fs::write(&bonds_file, bonds_content).unwrap();

        let result = BondsParser::parse(&bonds_file);
        assert!(result.is_ok());

        let bonds = result.unwrap();
        assert_eq!(bonds.len(), 2);
    }
}
