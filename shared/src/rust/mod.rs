pub mod dag;
pub mod grpc;
pub mod hashable_set;
pub mod shared;
pub mod store;

pub type ByteVector = Vec<u8>;
pub type ByteBuffer = Vec<u8>;
pub type Byte = u8;
pub type ByteString = Vec<u8>;
pub type BitSet = Vec<u8>;
pub type BitVector = Vec<u8>;

pub mod serde_bytes {
    use prost::bytes::Bytes;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert Bytes to &[u8] and serialize with serde_bytes
        ::serde_bytes::serialize(bytes.as_ref(), serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize to Vec<u8> and then convert to Bytes
        let bytes: Vec<u8> = ::serde_bytes::deserialize(deserializer)?;
        Ok(Bytes::from(bytes))
    }
}

pub mod serde_vec_bytes {
    use prost::bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes_vec: &Vec<Bytes>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert Vec<Bytes> to Vec<Vec<u8>> for serialization
        let vec_u8: Vec<Vec<u8>> = bytes_vec.iter().map(|b| b.to_vec()).collect();
        vec_u8.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Bytes>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as Vec<Vec<u8>> and convert to Vec<Bytes>
        let vec_u8: Vec<Vec<u8>> = Vec::deserialize(deserializer)?;
        let bytes_vec = vec_u8.into_iter().map(Bytes::from).collect();
        Ok(bytes_vec)
    }
}

pub mod serde_btreemap_bytes_i64 {
    use prost::bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;
    use std::hash::Hash;

    // Helper struct for serializing Bytes as keys
    #[derive(Eq, Ord, PartialOrd)]
    struct BytesKey(Bytes);

    impl Hash for BytesKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.0.as_ref().hash(state);
        }
    }

    impl PartialEq for BytesKey {
        fn eq(&self, other: &Self) -> bool {
            self.0.as_ref() == other.0.as_ref()
        }
    }

    impl Serialize for BytesKey {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            ::serde_bytes::serialize(self.0.as_ref(), serializer)
        }
    }

    impl<'de> Deserialize<'de> for BytesKey {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let bytes: Vec<u8> = ::serde_bytes::deserialize(deserializer)?;
            Ok(BytesKey(Bytes::from(bytes)))
        }
    }

    pub fn serialize<S>(map: &BTreeMap<Bytes, i64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert BTreeMap<Bytes, i64> to BTreeMap<BytesKey, i64> for serialization
        let transformed_map: BTreeMap<BytesKey, i64> =
            map.iter().map(|(k, v)| (BytesKey(k.clone()), *v)).collect();

        transformed_map.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<Bytes, i64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as BTreeMap<BytesKey, i64> and convert to BTreeMap<Bytes, i64>
        let transformed_map: BTreeMap<BytesKey, i64> = BTreeMap::deserialize(deserializer)?;
        let map = transformed_map.into_iter().map(|(k, v)| (k.0, v)).collect();

        Ok(map)
    }
}

pub mod serde_hex_bytes {
    use prost::bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Bytes, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert bytes to hex string (like Scala's PrettyPrinter.buildStringNoLimit)
        let hex_string = hex::encode(bytes.as_ref());
        hex_string.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Bytes, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hex_string: String = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_string)
            .map_err(|e| serde::de::Error::custom(format!("Invalid hex string: {}", e)))?;
        Ok(Bytes::from(bytes))
    }
}

pub mod serde_hex_vec_u8 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert bytes to hex string (like Scala's PrettyPrinter.buildStringNoLimit)
        let hex_string = hex::encode(bytes);
        hex_string.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hex_string: String = String::deserialize(deserializer)?;
        hex::decode(&hex_string)
            .map_err(|e| serde::de::Error::custom(format!("Invalid hex string: {}", e)))
    }
}

pub mod serde_always_equal_bitset {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(_: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as unit (like Scala's AlwaysEqual encoder)
        serializer.serialize_unit()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize unit and return empty BitSet
        let _: () = <()>::deserialize(deserializer)?;
        Ok(Vec::new())
    }
}
