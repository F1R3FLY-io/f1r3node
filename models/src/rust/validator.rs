// See models/src/main/scala/coop/rchain/models/Validator.scala
use serde::{Deserialize, Serialize};
use shared::rust::serde_bytes;

pub type Validator = prost::bytes::Bytes;

pub const LENGTH: usize = 65;

#[derive(Default, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ValidatorSerde(pub Validator);

impl From<Validator> for ValidatorSerde {
    fn from(validator: Validator) -> Self {
        ValidatorSerde(validator)
    }
}

impl From<ValidatorSerde> for Validator {
    fn from(wrapper: ValidatorSerde) -> Self {
        wrapper.0
    }
}

impl Serialize for ValidatorSerde {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(&self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for ValidatorSerde {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(ValidatorSerde(serde_bytes::deserialize(deserializer)?))
    }
}
