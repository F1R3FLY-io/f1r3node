// See crypto/src/test/scala/coop/rchain/crypto/util/DERConverterSpec.scala

use crypto::rust::hash::blake2b256::Blake2b256;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::secp256k1_eth::Secp256k1Eth;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;
use crypto::rust::util::certificate_helper::CertificateHelper;
use proptest::prelude::*;

/// Property-based test: DER converter check with valid and non-empty input
#[cfg(test)]
mod der_converter_tests {
    use super::*;

    proptest! {
        #[test]
        fn test_der_converter_with_valid_non_empty_input(bytes in prop::collection::vec(any::<u8>(), 1..1000)) {
            // Only test with non-empty input
            if !bytes.is_empty() {
                // Generate a new key pair
                let secp256k1 = Secp256k1;
                let (private_key, _public_key) = secp256k1.new_key_pair();

                // Hash the input data
                let data = Blake2b256::hash(bytes.clone());

                // Sign the data using ETH style
                let secp256k1_eth = Secp256k1Eth;
                let sig_rs = secp256k1_eth.sign(&data, &private_key.bytes);

                // Only proceed if signature was generated successfully
                if !sig_rs.is_empty() && sig_rs.len() == 64 {
                    // Encode to DER format
                    let sig_der = CertificateHelper::encode_signature_rs_to_der(&sig_rs)
                        .expect("DER encoding should succeed for valid signature");

                    // Decode back to R,S format
                    let expected_rs = CertificateHelper::decode_signature_der_to_rs(&sig_der)
                        .expect("DER decoding should succeed for valid DER");

                    // Encode / decode should get initial input for valid signature
                    prop_assert_eq!(sig_rs, expected_rs, "Roundtrip encoding/decoding should preserve signature");
                }

                // Encoder is safe of exception for any input
                let encoded_bytes = CertificateHelper::encode_signature_rs_to_der(&bytes);
                match encoded_bytes {
                    Ok(der_bytes) => {
                        prop_assert!(!der_bytes.is_empty(), "Encoded DER should not be empty when successful");
                    }
                    Err(_) => {
                        // Encoder may fail for invalid input, which is acceptable
                    }
                }

                // Decoder should fail with invalid DER message format
                let decode_result = CertificateHelper::decode_signature_der_to_rs(&bytes);
                prop_assert!(decode_result.is_err(), "Decoder should fail on arbitrary bytes");
            }
        }
    }

    /// Test: encoder should throw exception on empty input
    #[test]
    fn test_encoder_fails_on_empty_input() {
        let empty_bytes: Vec<u8> = vec![];
        let result = CertificateHelper::encode_signature_rs_to_der(&empty_bytes);

        assert!(result.is_err(), "Encoder should fail on empty input");

        // Verify the error message mentions empty input
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("empty"),
            "Error should mention empty input"
        );
    }

    /// Test: decoder should throw exception on empty input  
    #[test]
    fn test_decoder_fails_on_empty_input() {
        let empty_bytes: Vec<u8> = vec![];
        let result = CertificateHelper::decode_signature_der_to_rs(&empty_bytes);

        assert!(result.is_err(), "Decoder should fail on empty input");

        // Verify the error message mentions empty input
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("empty"),
            "Error should mention empty input"
        );
    }

    /// Test: roundtrip encoding/decoding with known valid signatures
    #[test]
    fn test_known_signature_roundtrip() {
        // Test with a known 64-byte signature (32-byte R + 32-byte S)
        let test_signature = vec![
            // R component (32 bytes)
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98,
            0x76, 0x54, 0x32, 0x10, // S component (32 bytes)
            0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45,
            0x23, 0x01, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0xef, 0xcd, 0xab, 0x89,
            0x67, 0x45, 0x23, 0x01,
        ];

        // Encode to DER
        let der_result = CertificateHelper::encode_signature_rs_to_der(&test_signature);
        assert!(
            der_result.is_ok(),
            "DER encoding should succeed for valid 64-byte signature"
        );

        let der_bytes = der_result.unwrap();
        assert!(!der_bytes.is_empty(), "DER bytes should not be empty");

        // Decode back to R,S
        let rs_result = CertificateHelper::decode_signature_der_to_rs(&der_bytes);
        assert!(
            rs_result.is_ok(),
            "DER decoding should succeed for valid DER"
        );

        let decoded_rs = rs_result.unwrap();
        assert_eq!(
            decoded_rs.len(),
            64,
            "Decoded signature should be exactly 64 bytes"
        );

        // The roundtrip should preserve the signature data
        // Note: Due to DER normalization, the exact bytes might differ, but the mathematical values should be preserved
        assert_eq!(
            decoded_rs, test_signature,
            "Roundtrip should preserve signature data"
        );
    }

    /// Test: decoder fails on invalid DER format
    #[test]
    fn test_decoder_fails_on_invalid_der() {
        let invalid_der_data = vec![
            0x30, 0x45, 0x02, 0x20, // Invalid DER sequence
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff,
            // Missing S component
        ];

        let result = CertificateHelper::decode_signature_der_to_rs(&invalid_der_data);
        assert!(result.is_err(), "Decoder should fail on invalid DER format");
    }

    /// Test: encoder handles edge case inputs
    #[test]
    fn test_encoder_edge_cases() {
        // Test with exactly 64 bytes (minimum valid signature size)
        let min_signature = vec![0x01; 64];
        let result = CertificateHelper::encode_signature_rs_to_der(&min_signature);
        assert!(result.is_ok(), "Encoder should handle 64-byte input");

        // Test with more than 64 bytes (should only use first 64)
        let oversized_signature = vec![0x01; 128];
        let result = CertificateHelper::encode_signature_rs_to_der(&oversized_signature);
        assert!(
            result.is_ok(),
            "Encoder should handle oversized input by using first 64 bytes"
        );

        // Test with less than 64 bytes (should fail)
        let undersized_signature = vec![0x01; 32];
        let result = CertificateHelper::encode_signature_rs_to_der(&undersized_signature);
        assert!(result.is_err(), "Encoder should fail on undersized input");
    }
}
