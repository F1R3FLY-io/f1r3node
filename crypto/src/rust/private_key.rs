// See crypto/src/main/scala/coop/rchain/crypto/PrivateKey.scala

use std::cmp::PartialEq;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Eq)]
pub struct PrivateKey {
    pub bytes: prost::bytes::Bytes,
}

impl PrivateKey {
    pub fn new(bytes: prost::bytes::Bytes) -> Self {
        PrivateKey { bytes }
    }

    pub fn from_bytes(bs: &[u8]) -> Self {
        PrivateKey::new(bs.to_vec().into())
    }
}

impl PartialEq for PrivateKey {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Hash for PrivateKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}
