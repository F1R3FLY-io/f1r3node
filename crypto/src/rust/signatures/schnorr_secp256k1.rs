use super::signatures_alg::SignaturesAlg;
use crate::rust::{hash::blake2b256::Blake2b256, private_key::PrivateKey, public_key::PublicKey};
use k256::elliptic_curve::rand_core::OsRng;
use k256::schnorr::{
    signature::hazmat::{PrehashSigner, PrehashVerifier},
    Signature, SigningKey, VerifyingKey,
};

pub const SCHNORR_SECP256K1_ALGORITHM_NAME: &str = "schnorr-secp256k1";
pub const SCHNORR_SECP256K1_SIGNING_DOMAIN: &[u8] = b"f1r3node/schnorr-secp256k1/signing/v1";
pub const SCHNORR_SECP256K1_ACCOUNT_DOMAIN: &[u8] = b"f1r3node/schnorr-secp256k1/account/v1";

const SCHNORR_PUBKEY_LEN: usize = 32;
const SCHNORR_SECRET_KEY_LEN: usize = 32;
const SCHNORR_SIG_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SchnorrSecp256k1;

impl SchnorrSecp256k1 {
    pub fn name() -> String {
        SCHNORR_SECP256K1_ALGORITHM_NAME.to_string()
    }

    pub fn signing_preimage(serialized_payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            SCHNORR_SECP256K1_SIGNING_DOMAIN.len() + 8 + serialized_payload.len(),
        );
        out.extend_from_slice(SCHNORR_SECP256K1_SIGNING_DOMAIN);
        out.extend_from_slice(&(serialized_payload.len() as u64).to_be_bytes());
        out.extend_from_slice(serialized_payload);
        out
    }

    pub fn domain_separated_hash(serialized_payload: &[u8]) -> Vec<u8> {
        Blake2b256::hash(Self::signing_preimage(serialized_payload))
    }

    pub fn account_identifier_xonly(public_key_xonly: &[u8]) -> Result<[u8; 32], String> {
        if public_key_xonly.len() != SCHNORR_PUBKEY_LEN {
            return Err(format!(
                "Expected {}-byte x-only key, got {}",
                SCHNORR_PUBKEY_LEN,
                public_key_xonly.len()
            ));
        }

        let mut preimage =
            Vec::with_capacity(SCHNORR_SECP256K1_ACCOUNT_DOMAIN.len() + 8 + public_key_xonly.len());
        preimage.extend_from_slice(SCHNORR_SECP256K1_ACCOUNT_DOMAIN);
        preimage.extend_from_slice(&(public_key_xonly.len() as u64).to_be_bytes());
        preimage.extend_from_slice(public_key_xonly);
        let digest = Blake2b256::hash(preimage);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        Ok(out)
    }

    fn parse_verifying_key(bytes: &[u8]) -> Option<VerifyingKey> {
        if bytes.len() != SCHNORR_PUBKEY_LEN {
            return None;
        }
        VerifyingKey::from_bytes(bytes).ok()
    }

    fn parse_signature(bytes: &[u8]) -> Option<Signature> {
        if bytes.len() != SCHNORR_SIG_LEN {
            return None;
        }
        Signature::try_from(bytes).ok()
    }
}

impl SignaturesAlg for SchnorrSecp256k1 {
    fn verify(&self, data: &[u8], signature: &[u8], pub_key: &[u8]) -> bool {
        // Keep parity with existing signing stack: validators verify a 32-byte prehash.
        if data.len() != 32 {
            tracing::warn!(
                "schnorr-secp256k1.verify: expected 32-byte prehash, got {} bytes",
                data.len()
            );
            return false;
        }

        let Some(vk) = Self::parse_verifying_key(pub_key) else {
            return false;
        };
        let Some(sig) = Self::parse_signature(signature) else {
            return false;
        };
        vk.verify_prehash(data, &sig).is_ok()
    }

    fn sign(&self, data: &[u8], sec: &[u8]) -> Vec<u8> {
        if data.len() != 32 {
            tracing::warn!(
                "schnorr-secp256k1.sign: expected 32-byte prehash, got {} bytes",
                data.len()
            );
            return Vec::new();
        }
        if sec.len() != SCHNORR_SECRET_KEY_LEN {
            tracing::warn!(
                "schnorr-secp256k1.sign: expected 32-byte secret key, got {} bytes",
                sec.len()
            );
            return Vec::new();
        }

        let Ok(signing_key) = SigningKey::from_bytes(sec) else {
            return Vec::new();
        };
        match signing_key.sign_prehash(data) {
            Ok(sig) => sig.to_bytes().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    fn to_public(&self, sec: &PrivateKey) -> PublicKey {
        let Ok(signing_key) = SigningKey::from_bytes(&sec.bytes) else {
            return PublicKey::from_bytes(&[]);
        };
        PublicKey::from_bytes(&signing_key.verifying_key().to_bytes())
    }

    fn new_key_pair(&self) -> (PrivateKey, PublicKey) {
        let signing_key = SigningKey::random(&mut OsRng);
        let private_key = PrivateKey::from_bytes(&signing_key.to_bytes());
        let public_key = PublicKey::from_bytes(&signing_key.verifying_key().to_bytes());
        (private_key, public_key)
    }

    fn name(&self) -> String {
        Self::name()
    }

    fn sig_length(&self) -> usize {
        SCHNORR_SIG_LEN
    }

    fn eq(&self, other: &dyn SignaturesAlg) -> bool {
        self.name() == other.name()
    }

    fn box_clone(&self) -> Box<dyn SignaturesAlg> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::hash::blake2b256::Blake2b256;

    fn decode_hex(hex_str: &str) -> Vec<u8> {
        hex::decode(hex_str).expect("valid hex")
    }

    #[test]
    fn generates_valid_key_pairs() {
        let alg = SchnorrSecp256k1;
        for _ in 0..100 {
            let (sk, pk) = alg.new_key_pair();
            assert_eq!(sk.bytes.len(), 32);
            assert_eq!(pk.bytes.len(), 32);
        }
    }

    #[test]
    fn signs_and_verifies_prehash() {
        let alg = SchnorrSecp256k1;
        let (sk, pk) = alg.new_key_pair();
        let msg = Blake2b256::hash(b"hello schnorr".to_vec());
        let sig = alg.sign(&msg, &sk.bytes);
        assert_eq!(sig.len(), 64);
        assert!(alg.verify(&msg, &sig, &pk.bytes));
    }

    #[test]
    fn rejects_wrong_domain_hash() {
        let alg = SchnorrSecp256k1;
        let (sk, pk) = alg.new_key_pair();
        let payload = b"same payload";
        let schnorr_hash = SchnorrSecp256k1::domain_separated_hash(payload);
        let legacy_hash = Blake2b256::hash(payload.to_vec());
        let sig = alg.sign(&schnorr_hash, &sk.bytes);

        assert!(alg.verify(&schnorr_hash, &sig, &pk.bytes));
        assert!(!alg.verify(&legacy_hash, &sig, &pk.bytes));
    }

    #[test]
    fn rejects_malformed_key_and_signature_lengths() {
        let alg = SchnorrSecp256k1;
        let msg = [0u8; 32];
        let bad_sig = vec![0u8; 63];
        let bad_pk = vec![0u8; 31];
        assert!(!alg.verify(&msg, &bad_sig, &bad_pk));
    }

    #[test]
    fn bip340_vector_zero_matches() {
        // Source: BIP340 test vector #0 (also used in k256 tests).
        let sk_bytes =
            decode_hex("0000000000000000000000000000000000000000000000000000000000000003");
        let expected_pk =
            decode_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
        let expected_sig = decode_hex(
            "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA821525F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0",
        );
        let msg = [0u8; 32];
        let aux = [0u8; 32];

        let sk = SigningKey::from_bytes(&sk_bytes).expect("valid sk");
        let sig = sk
            .sign_prehash_with_aux_rand(&msg, &aux)
            .expect("vector must sign");
        let pk = sk.verifying_key().to_bytes().to_vec();

        assert_eq!(pk, expected_pk);
        assert_eq!(sig.to_bytes().to_vec(), expected_sig);

        let vk = VerifyingKey::from_bytes(&expected_pk).expect("valid vk");
        assert!(vk.verify_prehash(&msg, &sig).is_ok());
    }

    #[test]
    fn builds_account_identifier_from_xonly_pubkey() {
        let alg = SchnorrSecp256k1;
        let (_sk, pk) = alg.new_key_pair();
        let account = SchnorrSecp256k1::account_identifier_xonly(&pk.bytes).expect("account id");
        assert_eq!(account.len(), 32);
    }
}
