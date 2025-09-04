use super::signatures_alg::SignaturesAlg;
use crate::rust::{private_key::PrivateKey, public_key::PublicKey};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;

// See crypto/src/main/scala/coop/rchain/crypto/signatures/Ed25519.scala
#[derive(Clone, Debug, PartialEq)]
pub struct Ed25519;

impl SignaturesAlg for Ed25519 {
    fn verify(&self, data: &[u8], signature: &[u8], pub_key: &[u8]) -> bool {
        let public_key = match parse_public_key(pub_key) {
            Ok(key) => key,
            Err(err) => {
                eprintln!("{}", err);
                return false;
            }
        };

        let signature = match parse_signature(signature) {
            Ok(sig) => sig,
            Err(err) => {
                eprintln!("{}", err);
                return false;
            }
        };

        public_key.verify(data, &signature).is_ok()
    }

    fn sign(&self, data: &[u8], sec: &[u8]) -> Vec<u8> {
        let signing_key = parse_signing_key(sec).expect("Secret key must be 32 bytes");
        let signature = signing_key.sign(data);
        signature.to_bytes().to_vec()
    }

    fn to_public(&self, sec: &PrivateKey) -> PublicKey {
        let signing_key = parse_signing_key(&sec.bytes).expect("Secret key must be 32 bytes");
        let public_key = signing_key.verifying_key();

        PublicKey::from_bytes(&public_key.to_bytes())
    }

    fn new_key_pair(&self) -> (PrivateKey, PublicKey) {
        let mut csprng = OsRng;

        let signing_key = SigningKey::generate(&mut csprng);
        let public_key = signing_key.verifying_key();

        let private_key = PrivateKey::from_bytes(&signing_key.to_bytes());
        let public_key = PublicKey::from_bytes(&public_key.to_bytes());

        (private_key, public_key)
    }

    fn name(&self) -> String {
        "ed25519".to_string()
    }

    fn sig_length(&self) -> usize {
        64
    }

    fn eq(&self, other: &dyn SignaturesAlg) -> bool {
        self.name() == other.name()
    }

    fn box_clone(&self) -> Box<dyn SignaturesAlg> {
        Box::new(self.clone())
    }
}

fn parse_public_key(pub_key: &[u8]) -> Result<VerifyingKey, &'static str> {
    pub_key
        .try_into()
        .map_err(|_| "Public key must be 32 bytes")
        .and_then(|key| VerifyingKey::from_bytes(key).map_err(|_| "Invalid public key"))
}

fn parse_signature(signature: &[u8]) -> Result<Signature, &'static str> {
    signature
        .try_into()
        .map_err(|_| "Signature must be 64 bytes")
        .map(Signature::from_bytes)
}

fn parse_signing_key(secret: &[u8]) -> Result<SigningKey, &'static str> {
    secret
        .try_into()
        .map_err(|_| "Secret key must be 32 bytes")
        .map(SigningKey::from_bytes)
}

// crypto/src/test/scala/coop/rchain/crypto/signatures/Ed25519Test.scala
#[cfg(test)]
mod tests {
    use super::Ed25519;
    use crate::rust::hash::keccak256::Keccak256;
    use crate::rust::private_key::PrivateKey;
    use crate::rust::signatures::signatures_alg::SignaturesAlg;
    use hex::{decode, encode};

    #[test]
    fn computes_public_key_from_secret_key() {
        let ed25519 = Ed25519;

        let sec =
            decode("b18e1d0045995ec3d010c387ccfeb984d783af8fbb0f40fa7db126d889f6dadd").unwrap();
        let expected_pub = "77f48b59caeda77751ed138b0ec667ff50f8768c25d48309a8f386a2bad187fb";

        let private_key = PrivateKey::from_bytes(&sec);
        let public_key = ed25519.to_public(&private_key);

        assert_eq!(hex::encode(public_key.bytes), expected_pub);
    }

    #[test]
    fn verifies_the_given_signature() {
        let ed25519 = Ed25519;

        let data = decode("916c7d1d268fc0e77c1bef238432573c39be577bbea0998936add2b50a653171ce18a542b0b7f96c1691a3be6031522894a8634183eda38798a0c5d5d79fbd01dd04a8646d71873b77b221998a81922d8105f892316369d5224c9983372d2313c6b1f4556ea26ba49d46e8b561e0fc76633ac9766e68e21fba7edca93c4c7460376d7f3ac22ff372c18f613f2ae2e856af40").unwrap();
        let sig = decode("6bd710a368c1249923fc7a1610747403040f0cc30815a00f9ff548a896bbda0b4eb2ca19ebcf917f0f34200a9edbad3901b64ab09cc5ef7b9bcc3c40c0ff7509").unwrap();
        let pub_key =
            decode("77f48b59caeda77751ed138b0ec667ff50f8768c25d48309a8f386a2bad187fb").unwrap();

        assert!(ed25519.verify(&data, &sig, &pub_key));
    }

    #[test]
    fn creates_a_signature() {
        let ed25519 = Ed25519;

        let data = decode("916c7d1d268fc0e77c1bef238432573c39be577bbea0998936add2b50a653171ce18a542b0b7f96c1691a3be6031522894a8634183eda38798a0c5d5d79fbd01dd04a8646d71873b77b221998a81922d8105f892316369d5224c9983372d2313c6b1f4556ea26ba49d46e8b561e0fc76633ac9766e68e21fba7edca93c4c7460376d7f3ac22ff372c18f613f2ae2e856af40").unwrap();
        let expected_sig = decode("6bd710a368c1249923fc7a1610747403040f0cc30815a00f9ff548a896bbda0b4eb2ca19ebcf917f0f34200a9edbad3901b64ab09cc5ef7b9bcc3c40c0ff7509").unwrap();
        let sec =
            decode("b18e1d0045995ec3d010c387ccfeb984d783af8fbb0f40fa7db126d889f6dadd").unwrap();

        let sig = ed25519.sign(&data, &sec);

        assert_eq!(sig, expected_sig);
    }

    //See rholang/examples/tut-verify-channel.md
    #[test]
    fn test_ed25519_integration_with_keccak256() {
        let ed25519 = Ed25519;

        // 1. Get hash
        let program = r#"
        new x, y, stdout(`rho:io:stdout`) in { 
           x!(@"name"!("Joe") | @"age"!(40)) |
           for (@r <- x) { @"keccak256Hash"!(r.toByteArray(), *y) } |
           for (@h <- y) { stdout!(h) }
        }
    "#;

        let hash = Keccak256::hash(program.as_bytes().to_vec());
        let hash_hex = encode(&hash);
        println!("Program hash: {}", hash_hex);

        // 2. Generate keys
        let (private_key, public_key) = ed25519.new_key_pair();
        println!("Private Key: {}", encode(&private_key.bytes));
        println!("Public Key: {}", encode(&public_key.bytes));

        // 3. Signing the hash
        let signature = ed25519.sign(&hash, &private_key.bytes);
        let signature_hex = encode(&signature);
        println!("Signature: {}", signature_hex);

        // Hash length parity check
        assert_eq!(hash_hex.len() % 2, 0, "Hash length must be even");
        assert_eq!(signature_hex.len() % 2, 0, "Signature length must be even");

        // 4. Signature verification
        let is_valid = ed25519.verify(&hash, &signature, &public_key.bytes);
        assert!(is_valid, "The signature should be valid");

        // 5. Validation with incorrect hash
        let corrupted_hash_hex = "a6da46a1dc7ed615d4cd6472a736249a4d11142d160dbef9f20ae493de908c4e";
        assert_eq!(
            corrupted_hash_hex.len() % 2,
            0,
            "Corrupted hash length must be even"
        );
        let corrupted_hash = decode(corrupted_hash_hex).expect("Failed to decode corrupted hash");
        let corrupted_is_valid = ed25519.verify(&corrupted_hash, &signature, &public_key.bytes);
        assert!(
            !corrupted_is_valid,
            "The signature should be invalid for corrupted hash"
        );
    }
}
