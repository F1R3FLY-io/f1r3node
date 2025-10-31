//! JSON serialization/deserialization for BlockInfo and related types
//!
//! This module provides custom JSON serialization for protobuf-generated types
//! that don't have serde derives by default.

use models::casper::{BlockInfo, DeployInfo};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use utoipa::ToSchema;

use crate::rust::api::serde_types::{
    deploy_info::DeployInfoSerde, light_block_info::LightBlockInfoSerde,
};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BlockInfoSerde {
    pub block_info: LightBlockInfoSerde,
    pub deploys: Vec<DeployInfoSerde>,
}

impl From<BlockInfo> for BlockInfoSerde {
    fn from(block: BlockInfo) -> Self {
        Self {
            block_info: block.block_info.unwrap_or_default().into(),
            deploys: block
                .deploys
                .iter()
                .map(|d| DeployInfoSerde::from(d.clone()))
                .collect(),
        }
    }
}

impl From<BlockInfoSerde> for BlockInfo {
    fn from(json: BlockInfoSerde) -> Self {
        BlockInfo {
            block_info: Some(json.block_info.into()),
            deploys: json
                .deploys
                .into_iter()
                .map(|d| DeployInfo::from(d))
                .collect(),
        }
    }
}

pub fn serialize_block_info<S>(block: BlockInfo, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let json_block = BlockInfoSerde::from(block);
    json_block.serialize(serializer)
}

pub fn deserialize_block_info<'de, D>(deserializer: D) -> Result<BlockInfo, D::Error>
where
    D: Deserializer<'de>,
{
    let json_block = BlockInfoSerde::deserialize(deserializer)?;
    Ok(json_block.into())
}

impl Default for BlockInfoSerde {
    fn default() -> Self {
        Self {
            block_info: LightBlockInfoSerde::default(),
            deploys: Vec::new(),
        }
    }
}
