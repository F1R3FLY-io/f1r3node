use blake2::{Blake2b, Digest, digest::consts::U32};
use hex::ToHex;
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};

// See src/firefly/f1r3fly/rspace/src/main/scala/coop/rchain/rspace/hashing/Blake2b256Hash.scala
// The 'Hash' macro is needed here for test util function 'check_same_elements'
// The 'Arbitrary' macro is needed here for proptest in hot_store_spec.rs
// The 'Default' macro is needed here for hot_store_spec.rs
#[derive(Eq, Clone, Serialize, Deserialize, Hash, Arbitrary, Default)]
pub struct Blake2b256Hash(pub Vec<u8>);

pub const LENGTH: i64 = 32;

impl Blake2b256Hash {
    pub fn new(data: &[u8]) -> Self {
        let mut hasher = Blake2b::<U32>::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Blake2b256Hash(hash.to_vec())
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Blake2b256Hash(bytes)
    }

    pub fn from_bytes_prost(bytes: &prost::bytes::Bytes) -> Self {
        Blake2b256Hash(bytes.to_vec())
    }

    pub fn from_hex(hex: &str) -> Self {
        let bytes = hex::decode(hex).unwrap();
        Blake2b256Hash::from_bytes(bytes)
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.0.clone()
    }

    pub fn to_bytes_prost(&self) -> prost::bytes::Bytes {
        prost::bytes::Bytes::from(self.0.clone())
    }
}

impl Ord for Blake2b256Hash {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Blake2b256Hash {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Blake2b256Hash {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl std::fmt::Display for Blake2b256Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Blake2b256Hash({})", self.0.encode_hex::<String>())
    }
}

impl Debug for Blake2b256Hash {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl ToHex for Blake2b256Hash {
    fn encode_hex<T: FromIterator<char>>(&self) -> T {
        self.0.encode_hex()
    }

    fn encode_hex_upper<T: FromIterator<char>>(&self) -> T {
        self.0.encode_hex_upper()
    }
}
