// Port of node/src/main/scala/coop/rchain/node/encode/JsonEncoder.scala

use models::rhoapi::*;
use models::rust::{par_map::ParMap, par_set::ParSet};
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use serde::{Deserialize, Serialize};
use serde_json;

/// JSON encoder/decoder for RChain types, matching Scala JsonEncoder behavior
pub struct JsonEncoder;

impl JsonEncoder {
    /// Serialize any serializable type to JSON string
    pub fn to_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
        serde_json::to_string(value)
    }

    /// Serialize any serializable type to pretty JSON string
    pub fn to_json_pretty<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(value)
    }

    /// Deserialize JSON string to any deserializable type
    pub fn from_json<'a, T: Deserialize<'a>>(json: &'a str) -> Result<T, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize Par to JSON (matches Scala encodePar)
    pub fn encode_par(par: &Par) -> Result<String, serde_json::Error> {
        Self::to_json(par)
    }

    /// Deserialize Par from JSON (matches Scala decodePar)
    pub fn decode_par(json: &str) -> Result<Par, serde_json::Error> {
        Self::from_json(json)
    }

    /// Serialize ParSet to JSON as array (matches Scala encodeParSet)
    pub fn encode_par_set(par_set: &ParSet) -> Result<String, serde_json::Error> {
        Self::to_json(par_set)
    }

    /// Deserialize ParSet from JSON array (matches Scala decodeParSet)
    pub fn decode_par_set(json: &str) -> Result<ParSet, serde_json::Error> {
        Self::from_json(json)
    }

    /// Serialize ParMap to JSON as array of pairs (matches Scala encodeParMap)
    pub fn encode_par_map(par_map: &ParMap) -> Result<String, serde_json::Error> {
        Self::to_json(par_map)
    }

    /// Deserialize ParMap from JSON array of pairs (matches Scala decodeParMap)
    pub fn decode_par_map(json: &str) -> Result<ParMap, serde_json::Error> {
        Self::from_json(json)
    }

    /// Serialize Blake2b512Random to JSON as null (matches Scala encodeBlake2b512Random)
    pub fn encode_blake2b512_random(_random: &Blake2b512Random) -> Result<String, serde_json::Error> {
        Self::to_json(&())
    }

    /// Deserialize Blake2b512Random from JSON null (matches Scala decodeDummyBlake2b512Random)
    pub fn decode_blake2b512_random(json: &str) -> Result<Blake2b512Random, serde_json::Error> {
        let _: () = Self::from_json(json)?;
        Ok(Blake2b512Random::create_from_bytes(&[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::rust::utils::*;

    #[test]
    fn test_par_set_roundtrip() {
        let par1 = new_gint_par(1, Vec::new(), false);
        let par2 = new_gint_par(2, Vec::new(), false);
        let par_set = ParSet::create_from_vec(vec![par1, par2]);

        let json = JsonEncoder::encode_par_set(&par_set).unwrap();
        let decoded = JsonEncoder::decode_par_set(&json).unwrap();

        // ParSet should serialize as array and deserialize back
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert_eq!(par_set.ps.sorted_pars.len(), decoded.ps.sorted_pars.len());
    }

    #[test]
    fn test_par_map_roundtrip() {
        let key1 = new_gint_par(1, Vec::new(), false);
        let val1 = new_gstring_par("a".to_string(), Vec::new(), false);
        let key2 = new_gint_par(2, Vec::new(), false);
        let val2 = new_gstring_par("b".to_string(), Vec::new(), false);
        
        let par_map = ParMap::create_from_vec(vec![(key1, val1), (key2, val2)]);

        let json = JsonEncoder::encode_par_map(&par_map).unwrap();
        let decoded = JsonEncoder::decode_par_map(&json).unwrap();

        // ParMap should serialize as array and deserialize back
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert_eq!(par_map.ps.sorted_list.len(), decoded.ps.sorted_list.len());
    }

    #[test]
    fn test_blake2b512_random_roundtrip() {
        let random = Blake2b512Random::create_from_bytes(&[1, 2, 3]);

        let json = JsonEncoder::encode_blake2b512_random(&random).unwrap();
        let decoded = JsonEncoder::decode_blake2b512_random(&json).unwrap();

        // Blake2b512Random should serialize as null/unit
        assert_eq!(json, "null");
        // Decoded should be a valid Blake2b512Random (doesn't need to match original)
        assert_eq!(decoded, Blake2b512Random::create_from_bytes(&[1]));
    }

    #[test]
    fn test_par_roundtrip() {
        let par = new_gint_par(42, Vec::new(), false);

        let json = JsonEncoder::encode_par(&par).unwrap();
        let decoded = JsonEncoder::decode_par(&json).unwrap();

        // Basic Par serialization should work
        assert!(json.contains("42"));
        assert_eq!(par.sends.len(), decoded.sends.len());
        assert_eq!(par.receives.len(), decoded.receives.len());
        assert_eq!(par.exprs.len(), decoded.exprs.len());
    }
}
