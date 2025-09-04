// See models/src/main/scala/coop/rchain/models/block/StateHash.scala

use serde::{Deserialize, Serialize};

use shared::rust::serde_bytes;

pub type StateHash = prost::bytes::Bytes;

pub const LENGTH: usize = 32;

#[derive(Default, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StateHashSerde(pub StateHash);

impl From<StateHash> for StateHashSerde {
    fn from(state_hash: StateHash) -> Self {
        StateHashSerde(state_hash)
    }
}

impl From<StateHashSerde> for StateHash {
    fn from(wrapper: StateHashSerde) -> Self {
        wrapper.0
    }
}

impl Serialize for StateHashSerde {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for StateHashSerde {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(StateHashSerde(serde_bytes::deserialize(deserializer)?))
    }
}
