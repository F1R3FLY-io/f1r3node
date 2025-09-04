// See models/src/main/scala/coop/rchain/models/BlockHash.scala
use serde::{Deserialize, Serialize};
use shared::rust::serde_bytes;

pub type BlockHash = prost::bytes::Bytes;

pub const LENGTH: usize = 32;

#[derive(Default, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockHashSerde(pub BlockHash);

impl From<BlockHash> for BlockHashSerde {
    fn from(hash: BlockHash) -> Self {
        BlockHashSerde(hash)
    }
}

impl From<BlockHashSerde> for BlockHash {
    fn from(wrapper: BlockHashSerde) -> Self {
        wrapper.0
    }
}

impl Serialize for BlockHashSerde {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for BlockHashSerde {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(BlockHashSerde(serde_bytes::deserialize(deserializer)?))
    }
}
