//! JSON serialization/deserialization for SystemDeployInfoWithEventData and related types
//!
//! This module provides custom JSON serialization for protobuf types that don't have serde derives by default.

use models::casper::{
    BondInfo, CloseBlockSystemDeployDataProto, JustificationInfo, PeekProto, RejectedDeployInfo,
    ReportCommProto, ReportConsumeProto, ReportProduceProto, ReportProto, SingleReport,
    SlashSystemDeployDataProto, SystemDeployDataProto, SystemDeployInfoWithEventData,
};
use models::rhoapi::{BindPattern, ListParWithRandom, Par};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondInfoSerde {
    pub validator: String,
    pub stake: i64,
}

impl From<BondInfo> for BondInfoSerde {
    fn from(data: BondInfo) -> Self {
        Self {
            validator: data.validator,
            stake: data.stake,
        }
    }
}

impl From<BondInfoSerde> for BondInfo {
    fn from(data: BondInfoSerde) -> Self {
        Self {
            validator: data.validator,
            stake: data.stake,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JustificationInfoSerde {
    pub validator: String,
    pub latest_block_hash: String,
}

impl From<JustificationInfo> for JustificationInfoSerde {
    fn from(data: JustificationInfo) -> Self {
        Self {
            validator: data.validator,
            latest_block_hash: data.latest_block_hash,
        }
    }
}

impl From<JustificationInfoSerde> for JustificationInfo {
    fn from(data: JustificationInfoSerde) -> Self {
        Self {
            validator: data.validator,
            latest_block_hash: data.latest_block_hash,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedDeployInfoSerde {
    pub sig: String,
}

impl From<RejectedDeployInfo> for RejectedDeployInfoSerde {
    fn from(data: RejectedDeployInfo) -> Self {
        Self { sig: data.sig }
    }
}

impl From<RejectedDeployInfoSerde> for RejectedDeployInfo {
    fn from(data: RejectedDeployInfoSerde) -> Self {
        Self { sig: data.sig }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SlashSystemDeployDataSerde {
    pub invalid_block_hash: Vec<u8>,
    pub issuer_public_key: Vec<u8>,
}

impl From<SlashSystemDeployDataProto> for SlashSystemDeployDataSerde {
    fn from(data: SlashSystemDeployDataProto) -> Self {
        Self {
            invalid_block_hash: data.invalid_block_hash.to_vec(),
            issuer_public_key: data.issuer_public_key.to_vec(),
        }
    }
}

impl From<SlashSystemDeployDataSerde> for SlashSystemDeployDataProto {
    fn from(data: SlashSystemDeployDataSerde) -> Self {
        Self {
            invalid_block_hash: data.invalid_block_hash.into(),
            issuer_public_key: data.issuer_public_key.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CloseBlockSystemDeployDataSerde {}

impl From<CloseBlockSystemDeployDataProto> for CloseBlockSystemDeployDataSerde {
    fn from(_data: CloseBlockSystemDeployDataProto) -> Self {
        Self {}
    }
}

impl From<CloseBlockSystemDeployDataSerde> for CloseBlockSystemDeployDataProto {
    fn from(_data: CloseBlockSystemDeployDataSerde) -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum SystemDeployDataSerde {
    SlashSystemDeploy(SlashSystemDeployDataSerde),
    CloseBlockSystemDeploy(CloseBlockSystemDeployDataSerde),
}

impl From<SystemDeployDataProto> for SystemDeployDataSerde {
    fn from(data: SystemDeployDataProto) -> Self {
        match data.system_deploy {
            Some(models::casper::system_deploy_data_proto::SystemDeploy::SlashSystemDeploy(
                slash,
            )) => Self::SlashSystemDeploy(slash.into()),
            Some(
                models::casper::system_deploy_data_proto::SystemDeploy::CloseBlockSystemDeploy(
                    close,
                ),
            ) => Self::CloseBlockSystemDeploy(close.into()),
            None => Self::CloseBlockSystemDeploy(CloseBlockSystemDeployDataSerde {}),
        }
    }
}

impl From<SystemDeployDataSerde> for SystemDeployDataProto {
    fn from(data: SystemDeployDataSerde) -> Self {
        let system_deploy = match data {
            SystemDeployDataSerde::SlashSystemDeploy(slash) => Some(
                models::casper::system_deploy_data_proto::SystemDeploy::SlashSystemDeploy(
                    slash.into(),
                ),
            ),
            SystemDeployDataSerde::CloseBlockSystemDeploy(close) => Some(
                models::casper::system_deploy_data_proto::SystemDeploy::CloseBlockSystemDeploy(
                    close.into(),
                ),
            ),
        };
        Self { system_deploy }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct PeekProtoSerde {
    pub channel_index: i32,
}

impl From<PeekProto> for PeekProtoSerde {
    fn from(data: PeekProto) -> Self {
        Self {
            channel_index: data.channel_index,
        }
    }
}

impl From<PeekProtoSerde> for PeekProto {
    fn from(data: PeekProtoSerde) -> Self {
        Self {
            channel_index: data.channel_index,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ReportProduceSerde {
    pub channel: Option<Par>,
    pub data: Option<ListParWithRandom>,
}

impl From<ReportProduceProto> for ReportProduceSerde {
    fn from(data: ReportProduceProto) -> Self {
        Self {
            channel: data.channel,
            data: data.data,
        }
    }
}

impl From<ReportProduceSerde> for ReportProduceProto {
    fn from(data: ReportProduceSerde) -> Self {
        Self {
            channel: data.channel,
            data: data.data,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ReportConsumeSerde {
    pub channels: Vec<Par>,
    pub patterns: Vec<BindPattern>,
    pub peeks: Vec<PeekProtoSerde>,
}

impl From<ReportConsumeProto> for ReportConsumeSerde {
    fn from(data: ReportConsumeProto) -> Self {
        Self {
            channels: data.channels,
            patterns: data.patterns,
            peeks: data.peeks.into_iter().map(|p| p.into()).collect(),
        }
    }
}

impl From<ReportConsumeSerde> for ReportConsumeProto {
    fn from(data: ReportConsumeSerde) -> Self {
        Self {
            channels: data.channels,
            patterns: data.patterns,
            peeks: data.peeks.into_iter().map(|p| p.into()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ReportCommSerde {
    pub consume: Option<ReportConsumeSerde>,
    pub produces: Vec<ReportProduceSerde>,
}

impl From<ReportCommProto> for ReportCommSerde {
    fn from(data: ReportCommProto) -> Self {
        Self {
            consume: data.consume.map(|c| c.into()),
            produces: data.produces.into_iter().map(|p| p.into()).collect(),
        }
    }
}

impl From<ReportCommSerde> for ReportCommProto {
    fn from(data: ReportCommSerde) -> Self {
        Self {
            consume: data.consume.map(|c| c.into()),
            produces: data.produces.into_iter().map(|p| p.into()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum ReportProtoSerde {
    Produce(ReportProduceSerde),
    Consume(ReportConsumeSerde),
    Comm(ReportCommSerde),
}

impl From<ReportProto> for ReportProtoSerde {
    fn from(data: ReportProto) -> Self {
        match data.report {
            Some(models::casper::report_proto::Report::Produce(produce)) => {
                Self::Produce(produce.into())
            }
            Some(models::casper::report_proto::Report::Consume(consume)) => {
                Self::Consume(consume.into())
            }
            Some(models::casper::report_proto::Report::Comm(comm)) => Self::Comm(comm.into()),
            None => Self::Produce(ReportProduceSerde {
                channel: None,
                data: None,
            }),
        }
    }
}

impl From<ReportProtoSerde> for ReportProto {
    fn from(data: ReportProtoSerde) -> Self {
        let report = match data {
            ReportProtoSerde::Produce(produce) => Some(
                models::casper::report_proto::Report::Produce(produce.into()),
            ),
            ReportProtoSerde::Consume(consume) => Some(
                models::casper::report_proto::Report::Consume(consume.into()),
            ),
            ReportProtoSerde::Comm(comm) => {
                Some(models::casper::report_proto::Report::Comm(comm.into()))
            }
        };
        Self { report }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SingleReportSerde {
    pub events: Vec<ReportProtoSerde>,
}

impl From<SingleReport> for SingleReportSerde {
    fn from(data: SingleReport) -> Self {
        Self {
            events: data.events.into_iter().map(|e| e.into()).collect(),
        }
    }
}

impl From<SingleReportSerde> for SingleReport {
    fn from(data: SingleReportSerde) -> Self {
        Self {
            events: data.events.into_iter().map(|e| e.into()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SystemDeployInfoWithEventSerde {
    pub system_deploy: Option<SystemDeployDataSerde>,
    pub report: Vec<SingleReportSerde>,
}

impl From<SystemDeployInfoWithEventData> for SystemDeployInfoWithEventSerde {
    fn from(data: SystemDeployInfoWithEventData) -> Self {
        Self {
            system_deploy: data.system_deploy.map(|s| s.into()),
            report: data.report.into_iter().map(|r| r.into()).collect(),
        }
    }
}

impl From<SystemDeployInfoWithEventSerde> for SystemDeployInfoWithEventData {
    fn from(data: SystemDeployInfoWithEventSerde) -> Self {
        Self {
            system_deploy: data.system_deploy.map(|s| s.into()),
            report: data.report.into_iter().map(|r| r.into()).collect(),
        }
    }
}
