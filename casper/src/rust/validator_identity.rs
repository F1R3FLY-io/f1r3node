// See casper/src/main/scala/coop/rchain/casper/ValidatorIdentity.scala

use crypto::rust::{
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{
        secp256k1::Secp256k1,
        signatures_alg::{SignaturesAlg, SignaturesAlgFactory},
    },
};
use models::{casper::Signature, rust::casper::protocol::casper_message::BlockMessage};

use super::util::proto_util;

#[derive(Clone)]
pub struct ValidatorIdentity {
    pub public_key: PublicKey,
    pub private_key: PrivateKey,
    pub signature_algorithm: String,
}

impl ValidatorIdentity {
    pub fn new(private_key: &PrivateKey) -> Self {
        let public_key = Secp256k1.to_public(private_key);

        Self {
            public_key,
            private_key: private_key.clone(),
            signature_algorithm: Secp256k1.name(),
        }
    }

    pub fn signature(&self, data: &[u8]) -> Signature {
        let signature = SignaturesAlgFactory::apply(&self.signature_algorithm)
            .expect("Failed to apply signature algorithm")
            .sign_with_private_key(data, &self.private_key);

        Signature {
            public_key: self.public_key.bytes.clone(),
            algorithm: self.signature_algorithm.clone(),
            sig: signature.into(),
        }
    }

    pub fn sign_block(&self, block: &BlockMessage) -> BlockMessage {
        let mut new_block = block.clone();
        new_block.sig_algorithm = self.signature_algorithm.clone();
        new_block.sender = self.public_key.bytes.clone();

        let block_hash = proto_util::hash_block(&new_block);
        let sig = self.signature(block_hash.as_ref());

        new_block.sig = sig.sig;
        new_block.block_hash = block_hash;

        new_block
    }

    pub fn from_private_key_with_logging(priv_key: Option<&str>) -> Option<Self> {
        println!("priv_key = {:?}", priv_key);
        match priv_key {
            Some(priv_key) => Self::from_hex(priv_key),
            None => {
                log::warn!("No private key detected, cannot create validator identification.");
                None
            }
        }
    }

    fn from_hex(priv_key_hex: &str) -> Option<Self> {
        let decoded = hex::decode(priv_key_hex).unwrap();
        let private_key = PrivateKey::from_bytes(&decoded);

        Some(Self::new(&private_key))
    }
}
