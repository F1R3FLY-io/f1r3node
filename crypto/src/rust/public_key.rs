use std::cmp::PartialEq;
use std::hash::{Hash, Hasher};

// See crypto/src/main/scala/coop/rchain/crypto/PublicKey.scala
#[derive(Debug, Clone, Eq, serde::Serialize, serde::Deserialize)]
pub struct PublicKey {
    #[serde(with = "shared::rust::serde_bytes")]
    pub bytes: prost::bytes::Bytes,
}

impl PublicKey {
    pub fn new(bytes: prost::bytes::Bytes) -> Self {
        PublicKey { bytes }
    }

    pub fn from_bytes(bs: &[u8]) -> Self {
        PublicKey::new(bs.to_vec().into())
    }
}

impl PartialEq for PublicKey {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Hash for PublicKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}
