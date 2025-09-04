// See crypto/src/main/scala/coop/rchain/crypto/signatures/SignaturesAlg.scala

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

use crate::rust::{private_key::PrivateKey, public_key::PublicKey};

use super::{secp256k1::Secp256k1, secp256k1_eth::Secp256k1Eth};

pub trait SignaturesAlg: std::fmt::Debug + Send + Sync {
    fn verify(&self, data: &[u8], signature: &[u8], pub_key: &[u8]) -> bool;

    fn sign(&self, data: &[u8], sec: &[u8]) -> Vec<u8>;

    fn to_public(&self, sec: &PrivateKey) -> PublicKey;

    fn new_key_pair(&self) -> (PrivateKey, PublicKey);

    fn name(&self) -> String;

    fn verify_with_public_key(&self, data: &[u8], signature: &[u8], pub_key: &PublicKey) -> bool {
        self.verify(data, signature, &pub_key.bytes)
    }

    fn sign_with_private_key(&self, data: &[u8], sec: &PrivateKey) -> Vec<u8> {
        self.sign(data, &sec.bytes)
    }

    fn sig_length(&self) -> usize;

    fn eq(&self, other: &dyn SignaturesAlg) -> bool;

    fn box_clone(&self) -> Box<dyn SignaturesAlg>;
}

impl Clone for Box<dyn SignaturesAlg> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

impl PartialEq for Box<dyn SignaturesAlg> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Serialize for Box<dyn SignaturesAlg> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.name())
    }
}

impl<'de> Deserialize<'de> for Box<dyn SignaturesAlg> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SignaturesAlgVisitor;

        impl<'de> Visitor<'de> for SignaturesAlgVisitor {
            type Value = Box<dyn SignaturesAlg>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a known signature algorithm name")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    "secp256k1" => Ok(Box::new(Secp256k1)),
                    "secp256k1-eth" => Ok(Box::new(Secp256k1Eth)),
                    // "ed25519" => Ok(Box::new(Ed25519)),
                    _ => Err(de::Error::custom(format!("Unknown algorithm: {}", value))),
                }
            }
        }

        deserializer.deserialize_str(SignaturesAlgVisitor)
    }
}

pub struct SignaturesAlgFactory;

impl SignaturesAlgFactory {
    pub fn apply(name: &str) -> Option<Box<dyn SignaturesAlg>> {
        match name {
            // ed25519 signature algorithm is disabled
            // TODO: quick way to prevent use of ed25519 to sign deploys - OLD
            // https://rchain.atlassian.net/browse/RCHAIN-3560
            // case Ed25519.name => Some(Ed25519)
            "secp256k1" => Some(Box::new(Secp256k1)),
            "secp256k1-eth" => Some(Box::new(Secp256k1Eth)),
            _ => None,
        }
    }
}
