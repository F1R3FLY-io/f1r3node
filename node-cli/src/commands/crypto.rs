use crate::args::*;
use crate::error::{NodeCliError, Result};
use crate::utils::{print_key, print_success, CryptoUtils};
use std::fs;
use std::path::Path;

pub fn generate_public_key_command(args: &GeneratePublicKeyArgs) -> Result<()> {
    // Decode private key using crypto utils
    let secret_key = CryptoUtils::decode_private_key(&args.private_key)?;

    // Derive public key from private key
    let public_key = CryptoUtils::derive_public_key(&secret_key);

    // Serialize public key in the requested format
    let public_key_hex = CryptoUtils::serialize_public_key(&public_key, args.compressed);

    // Print the public key using output utils
    let key_type = if args.compressed {
        "compressed"
    } else {
        "uncompressed"
    };
    print_key(&format!("Public key ({})", key_type), &public_key_hex);

    Ok(())
}

pub fn generate_key_pair_command(args: &GenerateKeyPairArgs) -> Result<()> {
    // Generate a new random key pair
    let (secret_key, public_key) = CryptoUtils::generate_key_pair()?;

    // Serialize keys
    let private_key_hex = CryptoUtils::serialize_private_key(&secret_key);
    let public_key_hex = CryptoUtils::serialize_public_key(&public_key, args.compressed);

    if args.save {
        // Create output directory if it doesn't exist
        let output_dir = Path::new(&args.output_dir);
        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(|e| {
                NodeCliError::file_write_failed(
                    &output_dir.display().to_string(),
                    &format!("Failed to create directory: {}", e),
                )
            })?;
        }

        // Create filenames
        let private_key_file = output_dir.join("private_key.hex");
        let public_key_file = output_dir.join("public_key.hex");

        // Write keys to files
        fs::write(&private_key_file, &private_key_hex).map_err(|e| {
            NodeCliError::file_write_failed(&private_key_file.display().to_string(), &e.to_string())
        })?;

        fs::write(&public_key_file, &public_key_hex).map_err(|e| {
            NodeCliError::file_write_failed(&public_key_file.display().to_string(), &e.to_string())
        })?;

        print_success(&format!(
            "Private key saved to: {}",
            private_key_file.display()
        ));
        print_success(&format!(
            "Public key saved to: {}",
            public_key_file.display()
        ));
    } else {
        // Print the keys using output utils
        print_key("Private key", &private_key_hex);
        let key_type = if args.compressed {
            "compressed"
        } else {
            "uncompressed"
        };
        print_key(&format!("Public key ({})", key_type), &public_key_hex);
    }

    Ok(())
}

pub fn generate_rev_address_command(args: &GenerateRevAddressArgs) -> Result<()> {
    // Determine the public key to use
    let public_key_hex = if let Some(public_key_hex) = &args.public_key {
        // Use provided public key
        public_key_hex.clone()
    } else if let Some(private_key_hex) = &args.private_key {
        // Derive public key from private key
        let secret_key = CryptoUtils::decode_private_key(private_key_hex)?;
        let public_key = CryptoUtils::derive_public_key(&secret_key);
        // Use uncompressed format for REV address generation
        CryptoUtils::serialize_public_key(&public_key, false)
    } else {
        return Err(NodeCliError::config_missing_required(
            "Either --public-key or --private-key must be provided",
        ));
    };

    // Validate the public key
    if !CryptoUtils::is_valid_public_key(&public_key_hex) {
        return Err(NodeCliError::crypto_invalid_public_key(
            "Invalid public key format",
        ));
    }

    // Generate REV address
    let rev_address = CryptoUtils::generate_rev_address(&public_key_hex)?;

    // Print the result using output utils
    print_key("Public key", &public_key_hex);
    print_key("REV address", &rev_address);

    Ok(())
}
