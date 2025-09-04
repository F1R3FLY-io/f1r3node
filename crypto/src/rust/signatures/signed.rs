use crate::rust::{
    hash::{blake2b256::Blake2b256, keccak256::Keccak256},
    private_key::PrivateKey,
    public_key::PublicKey,
};

use super::{secp256k1_eth::Secp256k1Eth, signatures_alg::SignaturesAlg};

// See crypto/src/main/scala/coop/rchain/crypto/signatures/Signed.scala
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Signed<A> {
    pub data: A,
    pub pk: PublicKey,
    #[serde(with = "shared::rust::serde_bytes")]
    pub sig: prost::bytes::Bytes,
    pub sig_algorithm: Box<dyn SignaturesAlg>,
}

impl<A: std::fmt::Debug + serde::Serialize> Signed<A> {
    pub fn create(
        data: A,
        sig_algorithm: Box<dyn SignaturesAlg>,
        sk: PrivateKey,
    ) -> Result<Self, String> {
        let serialized_data =
            bincode::serialize(&data).map_err(|e| format!("Failed to serialize data: {}", e))?;
        let hash = Signed::<A>::signature_hash(&sig_algorithm.name(), serialized_data);
        let sig = sig_algorithm.sign(&hash, &sk.bytes);

        Ok(Self {
            data,
            pk: sig_algorithm.to_public(&sk),
            sig: prost::bytes::Bytes::from(sig),
            sig_algorithm,
        })
    }

    pub fn from_signed_data(
        data: A,
        pk: PublicKey,
        sig: prost::bytes::Bytes,
        sig_algorithm: Box<dyn SignaturesAlg>,
    ) -> Result<Option<Self>, String> {
        let serialized_data =
            bincode::serialize(&data).map_err(|e| format!("Failed to serialize data: {}", e))?;
        let hash = Signed::<A>::signature_hash(&sig_algorithm.name(), serialized_data);

        if sig_algorithm.verify(&hash, &sig, &pk.bytes) {
            Ok(Some(Self {
                data,
                pk,
                sig,
                sig_algorithm,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn signature_hash(sig_alg_name: &str, serialized_data: Vec<u8>) -> Vec<u8> {
        match sig_alg_name {
            name if name == Secp256k1Eth::name() => {
                let prefix = Signed::<A>::eth_prefix(serialized_data.len());
                let mut combined = prefix;
                combined.extend(serialized_data);
                Keccak256::hash(combined)
            }

            _ => Blake2b256::hash(serialized_data),
        }
    }

    fn eth_prefix(msg_length: usize) -> Vec<u8> {
        format!("\u{0019}Ethereum Signed Message:\n{}", msg_length)
            .as_bytes()
            .to_vec()
    }
}

impl<A: PartialEq> PartialEq for Signed<A> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.pk == other.pk
            && self.sig == other.sig
            && self.sig_algorithm.eq(&other.sig_algorithm)
    }
}

impl<A: Eq> Eq for Signed<A> {}

impl<A: std::hash::Hash> std::hash::Hash for Signed<A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
        self.pk.hash(state);
        self.sig.hash(state);
        self.sig_algorithm.name().hash(state);
    }
}
