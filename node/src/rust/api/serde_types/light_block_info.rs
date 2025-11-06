//! JSON serialization/deserialization for LightBlockInfo and related types
//!
//! This module provides custom JSON serialization for protobuf-generated types
//! that don't have serde derives by default.

use models::casper::{BondInfo, JustificationInfo, LightBlockInfo, RejectedDeployInfo};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use utoipa::ToSchema;

/// Serializable representation of LightBlockInfo
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LightBlockInfoSerde {
    pub block_hash: String,
    pub sender: String,
    pub seq_num: i64,
    pub sig: String,
    pub sig_algorithm: String,
    pub shard_id: String,
    pub extra_bytes: Vec<u8>, // Raw bytes
    pub version: i64,
    pub timestamp: i64,
    pub header_extra_bytes: Vec<u8>, // Raw bytes
    pub parents_hash_list: Vec<String>,
    pub block_number: i64,
    pub pre_state_hash: String,
    pub post_state_hash: String,
    pub body_extra_bytes: Vec<u8>, // Raw bytes
    pub bonds: Vec<BondInfoJson>,
    pub block_size: String,
    pub deploy_count: i32,
    pub fault_tolerance: f32,
    pub justifications: Vec<JustificationInfoJson>,
    pub rejected_deploys: Vec<RejectedDeployInfoJson>,
}

/// Custom JSON representation of BondInfo
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BondInfoJson {
    pub validator: String,
    pub stake: i64,
}

/// Custom JSON representation of JustificationInfo
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct JustificationInfoJson {
    pub validator: String,
    pub latest_block_hash: String,
}

/// Custom JSON representation of RejectedDeployInfo
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RejectedDeployInfoJson {
    pub sig: String,
}

/// Convert LightBlockInfo to JSON-serializable format
impl From<LightBlockInfo> for LightBlockInfoSerde {
    fn from(block: LightBlockInfo) -> Self {
        Self {
            block_hash: block.block_hash.clone(),
            sender: block.sender.clone(),
            seq_num: block.seq_num,
            sig: block.sig.clone(),
            sig_algorithm: block.sig_algorithm.clone(),
            shard_id: block.shard_id.clone(),
            extra_bytes: block.extra_bytes.to_vec(),
            version: block.version,
            timestamp: block.timestamp,
            header_extra_bytes: block.header_extra_bytes.to_vec(),
            parents_hash_list: block.parents_hash_list.clone(),
            block_number: block.block_number,
            pre_state_hash: block.pre_state_hash.clone(),
            post_state_hash: block.post_state_hash.clone(),
            body_extra_bytes: block.body_extra_bytes.to_vec(),
            bonds: block
                .bonds
                .iter()
                .map(|b| BondInfoJson {
                    validator: b.validator.clone(),
                    stake: b.stake,
                })
                .collect(),
            block_size: block.block_size.clone(),
            deploy_count: block.deploy_count,
            fault_tolerance: block.fault_tolerance,
            justifications: block
                .justifications
                .iter()
                .map(|j| JustificationInfoJson {
                    validator: j.validator.clone(),
                    latest_block_hash: j.latest_block_hash.clone(),
                })
                .collect(),
            rejected_deploys: block
                .rejected_deploys
                .iter()
                .map(|r| RejectedDeployInfoJson { sig: r.sig.clone() })
                .collect(),
        }
    }
}

/// Convert JSON format back to LightBlockInfo
impl From<LightBlockInfoSerde> for LightBlockInfo {
    fn from(json: LightBlockInfoSerde) -> Self {
        LightBlockInfo {
            block_hash: json.block_hash,
            sender: json.sender,
            seq_num: json.seq_num,
            sig: json.sig,
            sig_algorithm: json.sig_algorithm,
            shard_id: json.shard_id,
            extra_bytes: json.extra_bytes.into(),
            version: json.version,
            timestamp: json.timestamp,
            header_extra_bytes: json.header_extra_bytes.into(),
            parents_hash_list: json.parents_hash_list,
            block_number: json.block_number,
            pre_state_hash: json.pre_state_hash,
            post_state_hash: json.post_state_hash,
            body_extra_bytes: json.body_extra_bytes.into(),
            bonds: json
                .bonds
                .into_iter()
                .map(|b| BondInfo {
                    validator: b.validator,
                    stake: b.stake,
                })
                .collect(),
            block_size: json.block_size,
            deploy_count: json.deploy_count,
            fault_tolerance: json.fault_tolerance,
            justifications: json
                .justifications
                .into_iter()
                .map(|j| JustificationInfo {
                    validator: j.validator,
                    latest_block_hash: j.latest_block_hash,
                })
                .collect(),
            rejected_deploys: json
                .rejected_deploys
                .into_iter()
                .map(|r| RejectedDeployInfo { sig: r.sig })
                .collect(),
        }
    }
}

/// Custom serializer for LightBlockInfo
pub fn serialize_light_block_info<S>(
    block: LightBlockInfo,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let json_block = LightBlockInfoSerde::from(block);
    json_block.serialize(serializer)
}

/// Custom deserializer for LightBlockInfo
pub fn deserialize_light_block_info<'de, D>(deserializer: D) -> Result<LightBlockInfo, D::Error>
where
    D: Deserializer<'de>,
{
    let json_block = LightBlockInfoSerde::deserialize(deserializer)?;
    json_block.try_into().map_err(serde::de::Error::custom)
}

impl Default for LightBlockInfoSerde {
    fn default() -> Self {
        Self {
            block_hash: String::new(),
            sender: String::new(),
            seq_num: 0,
            sig: String::new(),
            sig_algorithm: String::new(),
            shard_id: String::new(),
            extra_bytes: Vec::new(),
            version: 0,
            timestamp: 0,
            header_extra_bytes: Vec::new(),
            parents_hash_list: Vec::new(),
            block_number: 0,
            pre_state_hash: String::new(),
            post_state_hash: String::new(),
            body_extra_bytes: Vec::new(),
            bonds: Vec::new(),
            block_size: String::new(),
            deploy_count: 0,
            fault_tolerance: 0.0,
            justifications: Vec::new(),
            rejected_deploys: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::bytes::Bytes;

    /// Helper function to serialize LightBlockInfo to JSON string
    fn light_block_info_to_json(block: LightBlockInfo) -> Result<String, serde_json::Error> {
        let json_block = LightBlockInfoSerde::from(block);
        serde_json::to_string(&json_block)
    }

    /// Helper function to deserialize LightBlockInfo from JSON string
    fn light_block_info_from_json(json: &str) -> LightBlockInfo {
        let json_block: LightBlockInfoSerde = serde_json::from_str(json)
            .map_err(|e| format!("JSON parsing error: {}", e))
            .unwrap();
        json_block.into()
    }

    fn create_test_light_block_info() -> LightBlockInfo {
        LightBlockInfo {
            block_hash: "block_hash_123".to_string(),
            sender: "sender_456".to_string(),
            seq_num: 1,
            sig: "sig_789".to_string(),
            sig_algorithm: "secp256k1".to_string(),
            shard_id: "shard_001".to_string(),
            extra_bytes: Bytes::from(vec![1, 2, 3]),
            version: 1,
            timestamp: 1234567890,
            header_extra_bytes: Bytes::from(vec![4, 5, 6]),
            parents_hash_list: vec!["parent1".to_string(), "parent2".to_string()],
            block_number: 100,
            pre_state_hash: "pre_state_hash".to_string(),
            post_state_hash: "post_state_hash".to_string(),
            body_extra_bytes: Bytes::from(vec![7, 8, 9]),
            bonds: vec![BondInfo {
                validator: "validator1".to_string(),
                stake: 1000,
            }],
            block_size: "1024".to_string(),
            deploy_count: 5,
            fault_tolerance: 0.5,
            justifications: vec![JustificationInfo {
                validator: "validator1".to_string(),
                latest_block_hash: "latest_hash".to_string(),
            }],
            rejected_deploys: vec![RejectedDeployInfo {
                sig: "rejected_sig".to_string(),
            }],
        }
    }

    #[test]
    fn test_light_block_info_serialization() {
        let original = create_test_light_block_info();

        // Test JSON serialization
        let json = light_block_info_to_json(original.clone()).unwrap();
        println!("Serialized JSON: {}", json);

        // Test JSON deserialization
        let deserialized = light_block_info_from_json(&json);

        // Verify all fields match
        assert_eq!(original.block_hash, deserialized.block_hash);
        assert_eq!(original.sender, deserialized.sender);
        assert_eq!(original.seq_num, deserialized.seq_num);
        assert_eq!(original.sig, deserialized.sig);
        assert_eq!(original.sig_algorithm, deserialized.sig_algorithm);
        assert_eq!(original.shard_id, deserialized.shard_id);
        assert_eq!(original.extra_bytes, deserialized.extra_bytes);
        assert_eq!(original.version, deserialized.version);
        assert_eq!(original.timestamp, deserialized.timestamp);
        assert_eq!(original.header_extra_bytes, deserialized.header_extra_bytes);
        assert_eq!(original.parents_hash_list, deserialized.parents_hash_list);
        assert_eq!(original.block_number, deserialized.block_number);
        assert_eq!(original.pre_state_hash, deserialized.pre_state_hash);
        assert_eq!(original.post_state_hash, deserialized.post_state_hash);
        assert_eq!(original.body_extra_bytes, deserialized.body_extra_bytes);
        assert_eq!(original.bonds.len(), deserialized.bonds.len());
        assert_eq!(original.bonds[0].validator, deserialized.bonds[0].validator);
        assert_eq!(original.bonds[0].stake, deserialized.bonds[0].stake);
        assert_eq!(original.block_size, deserialized.block_size);
        assert_eq!(original.deploy_count, deserialized.deploy_count);
        assert_eq!(original.fault_tolerance, deserialized.fault_tolerance);
        assert_eq!(
            original.justifications.len(),
            deserialized.justifications.len()
        );
        assert_eq!(
            original.justifications[0].validator,
            deserialized.justifications[0].validator
        );
        assert_eq!(
            original.justifications[0].latest_block_hash,
            deserialized.justifications[0].latest_block_hash
        );
        assert_eq!(
            original.rejected_deploys.len(),
            deserialized.rejected_deploys.len()
        );
        assert_eq!(
            original.rejected_deploys[0].sig,
            deserialized.rejected_deploys[0].sig
        );
    }
}
