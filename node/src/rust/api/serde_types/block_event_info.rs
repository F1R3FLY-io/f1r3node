use models::casper::BlockEventInfo;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::rust::api::serde_types::{
    deploy_info::DeployInfoWithEventDataSerde, light_block_info::LightBlockInfoSerde,
    system_deploy_info::SystemDeployInfoWithEventSerde,
};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BlockEventInfoSerde {
    pub block_info: Option<LightBlockInfoSerde>,
    pub deploys: Vec<DeployInfoWithEventDataSerde>,
    pub system_deploys: Vec<SystemDeployInfoWithEventSerde>,
    pub post_state_hash: Vec<u8>,
}

impl From<BlockEventInfo> for BlockEventInfoSerde {
    fn from(data: BlockEventInfo) -> Self {
        Self {
            block_info: data.block_info.map(|b| b.into()),
            deploys: data.deploys.into_iter().map(|d| d.into()).collect(),
            system_deploys: data.system_deploys.into_iter().map(|s| s.into()).collect(),
            post_state_hash: data.post_state_hash.to_vec(),
        }
    }
}

impl From<BlockEventInfoSerde> for BlockEventInfo {
    fn from(data: BlockEventInfoSerde) -> Self {
        Self {
            block_info: data.block_info.map(|b| b.into()),
            deploys: data.deploys.into_iter().map(|d| d.into()).collect(),
            system_deploys: data.system_deploys.into_iter().map(|s| s.into()).collect(),
            post_state_hash: data.post_state_hash.into(),
        }
    }
}
