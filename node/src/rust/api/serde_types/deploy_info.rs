//! JSON serialization/deserialization for DeployInfo
//!
//! This module provides custom JSON serialization for the DeployInfo protobuf type
//! that doesn't have serde derives by default.

use models::casper::{DeployInfo, DeployInfoWithEventData, TransferInfo};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::rust::api::serde_types::system_deploy_info::SingleReportSerde;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TransferInfoSerde {
    #[serde(rename = "fromAddr")]
    pub from_addr: String,
    #[serde(rename = "toAddr")]
    pub to_addr: String,
    pub amount: i64,
    pub success: bool,
    #[serde(rename = "failReason")]
    pub fail_reason: String,
}

impl From<TransferInfo> for TransferInfoSerde {
    fn from(t: TransferInfo) -> Self {
        Self {
            from_addr: t.from_addr,
            to_addr: t.to_addr,
            amount: t.amount,
            success: t.success,
            fail_reason: t.fail_reason,
        }
    }
}

impl From<TransferInfoSerde> for TransferInfo {
    fn from(t: TransferInfoSerde) -> Self {
        TransferInfo {
            from_addr: t.from_addr,
            to_addr: t.to_addr,
            amount: t.amount,
            success: t.success,
            fail_reason: t.fail_reason,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeployInfoSerde {
    pub deployer: String,
    pub term: String,
    pub timestamp: i64,
    pub sig: String,
    #[serde(rename = "sigAlgorithm")]
    pub sig_algorithm: String,
    #[serde(rename = "phloPrice")]
    pub phlo_price: i64,
    #[serde(rename = "phloLimit")]
    pub phlo_limit: i64,
    #[serde(rename = "validAfterBlockNumber")]
    pub valid_after_block_number: i64,
    pub cost: u64,
    pub errored: bool,
    #[serde(rename = "systemDeployError")]
    pub system_deploy_error: String,
    pub transfers: Vec<TransferInfoSerde>,
}

impl From<DeployInfo> for DeployInfoSerde {
    fn from(deploy: DeployInfo) -> Self {
        Self {
            deployer: deploy.deployer,
            term: deploy.term,
            timestamp: deploy.timestamp,
            sig: deploy.sig,
            sig_algorithm: deploy.sig_algorithm,
            phlo_price: deploy.phlo_price,
            phlo_limit: deploy.phlo_limit,
            valid_after_block_number: deploy.valid_after_block_number,
            cost: deploy.cost,
            errored: deploy.errored,
            system_deploy_error: deploy.system_deploy_error,
            transfers: deploy.transfers.into_iter().map(TransferInfoSerde::from).collect(),
        }
    }
}

impl From<DeployInfoSerde> for DeployInfo {
    fn from(json: DeployInfoSerde) -> Self {
        DeployInfo {
            deployer: json.deployer,
            term: json.term,
            timestamp: json.timestamp,
            sig: json.sig,
            sig_algorithm: json.sig_algorithm,
            phlo_price: json.phlo_price,
            phlo_limit: json.phlo_limit,
            valid_after_block_number: json.valid_after_block_number,
            cost: json.cost,
            errored: json.errored,
            system_deploy_error: json.system_deploy_error,
            transfers: json.transfers.into_iter().map(TransferInfo::from).collect(),
        }
    }
}

impl Default for DeployInfoSerde {
    fn default() -> Self {
        Self {
            deployer: String::new(),
            term: String::new(),
            timestamp: 0,
            sig: String::new(),
            sig_algorithm: String::new(),
            phlo_price: 0,
            phlo_limit: 0,
            valid_after_block_number: 0,
            cost: 0,
            errored: false,
            system_deploy_error: String::new(),
            transfers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeployInfoWithEventDataSerde {
    #[serde(rename = "deployInfo")]
    pub deploy_info: Option<DeployInfoSerde>,
    pub report: Vec<SingleReportSerde>,
}

impl From<DeployInfoWithEventData> for DeployInfoWithEventDataSerde {
    fn from(data: DeployInfoWithEventData) -> Self {
        Self {
            deploy_info: data.deploy_info.map(|d| d.into()),
            report: data.report.into_iter().map(|r| r.into()).collect(),
        }
    }
}

impl From<DeployInfoWithEventDataSerde> for DeployInfoWithEventData {
    fn from(data: DeployInfoWithEventDataSerde) -> Self {
        Self {
            deploy_info: data.deploy_info.map(|d| d.into()),
            report: data.report.into_iter().map(|r| r.into()).collect(),
        }
    }
}
