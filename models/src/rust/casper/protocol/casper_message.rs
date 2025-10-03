// See models/src/main/scala/coop/rchain/casper/protocol/CasperMessage.scala

use crypto::rust::{
    public_key::PublicKey,
    signatures::{signatures_alg::SignaturesAlgFactory, signed::Signed},
};
use prost::Message;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, state::rspace_exporter::RSpaceExporterInstance,
};
use shared::rust::{Byte, ByteVector};

use crate::{
    casper::{system_deploy_data_proto::SystemDeploy, *},
    rhoapi::PCost,
    rust::casper::pretty_printer::PrettyPrinter,
};

// TODO: Use type ByteString from models crate
type ByteString = prost::bytes::Bytes;

#[derive(Clone, Debug, PartialEq)]
pub enum CasperMessage {
    BlockHashMessage(BlockHashMessage),
    BlockMessage(BlockMessage),
    ApprovedBlockCandidate(ApprovedBlockCandidate),
    ApprovedBlock(ApprovedBlock),
    ApprovedBlockRequest(ApprovedBlockRequest),
    BlockApproval(BlockApproval),
    BlockRequest(BlockRequest),
    ForkChoiceTipRequest(ForkChoiceTipRequest),
    HasBlock(HasBlock),
    HasBlockRequest(HasBlockRequest),
    NoApprovedBlockAvailable(NoApprovedBlockAvailable),
    UnapprovedBlock(UnapprovedBlock),
    // Last finalized state messages
    StoreItemsMessageRequest(StoreItemsMessageRequest),
    StoreItemsMessage(StoreItemsMessage),
}

impl CasperMessage {
    /// Convert from individual proto message types to CasperMessage
    /// This matches the Scala CasperMessage.from method behavior
    pub fn from_block_hash_message(proto: BlockHashMessageProto) -> Self {
        CasperMessage::BlockHashMessage(BlockHashMessage::from_proto(proto))
    }

    pub fn from_block_message(proto: BlockMessageProto) -> Result<Self, String> {
        Ok(CasperMessage::BlockMessage(BlockMessage::from_proto(
            proto,
        )?))
    }

    pub fn from_approved_block_candidate(
        proto: ApprovedBlockCandidateProto,
    ) -> Result<Self, String> {
        Ok(CasperMessage::ApprovedBlockCandidate(
            ApprovedBlockCandidate::from_proto(proto)?,
        ))
    }

    pub fn from_approved_block(proto: ApprovedBlockProto) -> Result<Self, String> {
        Ok(CasperMessage::ApprovedBlock(ApprovedBlock::from_proto(
            proto,
        )?))
    }

    pub fn from_approved_block_request(proto: ApprovedBlockRequestProto) -> Self {
        CasperMessage::ApprovedBlockRequest(ApprovedBlockRequest::from_proto(proto))
    }

    pub fn from_block_approval(proto: BlockApprovalProto) -> Result<Self, String> {
        Ok(CasperMessage::BlockApproval(BlockApproval::from_proto(
            proto,
        )?))
    }

    pub fn from_block_request(proto: BlockRequestProto) -> Self {
        CasperMessage::BlockRequest(BlockRequest::from_proto(proto))
    }

    pub fn from_fork_choice_tip_request(_proto: ForkChoiceTipRequestProto) -> Self {
        CasperMessage::ForkChoiceTipRequest(ForkChoiceTipRequest)
    }

    pub fn from_has_block(proto: HasBlockProto) -> Self {
        CasperMessage::HasBlock(HasBlock::from_proto(proto))
    }

    pub fn from_has_block_request(proto: HasBlockRequestProto) -> Self {
        CasperMessage::HasBlockRequest(HasBlockRequest::from_proto(proto))
    }

    pub fn from_no_approved_block_available(proto: NoApprovedBlockAvailableProto) -> Self {
        CasperMessage::NoApprovedBlockAvailable(NoApprovedBlockAvailable::from_proto(proto))
    }

    pub fn from_unapproved_block(proto: UnapprovedBlockProto) -> Result<Self, String> {
        Ok(CasperMessage::UnapprovedBlock(UnapprovedBlock::from_proto(
            proto,
        )?))
    }

    pub fn from_store_items_message_request(proto: StoreItemsMessageRequestProto) -> Self {
        CasperMessage::StoreItemsMessageRequest(StoreItemsMessageRequest::from_proto(proto))
    }

    pub fn from_store_items_message(proto: StoreItemsMessageProto) -> Self {
        CasperMessage::StoreItemsMessage(StoreItemsMessage::from_proto(proto))
    }
}

// TODO: Remove all into() and to_vec() once we have correct ByteString type in the models crate
#[derive(Clone, Debug, PartialEq)]
pub struct HasBlockRequest {
    pub hash: ByteString,
}

impl HasBlockRequest {
    pub fn from_proto(proto: HasBlockRequestProto) -> Self {
        Self { hash: proto.hash }
    }

    pub fn to_proto(self) -> HasBlockRequestProto {
        HasBlockRequestProto { hash: self.hash }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HasBlock {
    pub hash: ByteString,
}

impl HasBlock {
    pub fn from_proto(proto: HasBlockProto) -> Self {
        Self { hash: proto.hash }
    }

    pub fn to_proto(self) -> HasBlockProto {
        HasBlockProto { hash: self.hash }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockRequest {
    pub hash: ByteString,
}

impl BlockRequest {
    pub fn from_proto(proto: BlockRequestProto) -> Self {
        Self { hash: proto.hash }
    }

    pub fn to_proto(self) -> BlockRequestProto {
        BlockRequestProto { hash: self.hash }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForkChoiceTipRequest;

impl ForkChoiceTipRequest {
    pub fn to_proto(self) -> ForkChoiceTipRequestProto {
        ForkChoiceTipRequestProto {}
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApprovedBlockCandidate {
    pub block: BlockMessage,
    pub required_sigs: i32,
}

impl ApprovedBlockCandidate {
    pub fn from_proto(proto: ApprovedBlockCandidateProto) -> Result<Self, String> {
        Ok(Self {
            block: BlockMessage::from_proto(
                proto
                    .block
                    .ok_or_else(|| "Missing block field".to_string())?,
            )?,
            required_sigs: proto.required_sigs,
        })
    }

    pub fn to_proto(self) -> ApprovedBlockCandidateProto {
        ApprovedBlockCandidateProto {
            block: Some(self.block.to_proto()),
            required_sigs: self.required_sigs,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnapprovedBlock {
    pub candidate: ApprovedBlockCandidate,
    pub timestamp: i64,
    pub duration: i64,
}

impl UnapprovedBlock {
    pub fn from_proto(proto: UnapprovedBlockProto) -> Result<Self, String> {
        Ok(Self {
            candidate: ApprovedBlockCandidate::from_proto(
                proto
                    .candidate
                    .ok_or_else(|| "Missing candidate field".to_string())?,
            )?,
            timestamp: proto.timestamp,
            duration: proto.duration,
        })
    }

    pub fn to_proto(self) -> UnapprovedBlockProto {
        UnapprovedBlockProto {
            candidate: Some(self.candidate.to_proto()),
            timestamp: self.timestamp,
            duration: self.duration,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockApproval {
    pub candidate: ApprovedBlockCandidate,
    pub sig: Signature,
}

impl BlockApproval {
    pub fn from_proto(proto: BlockApprovalProto) -> Result<Self, String> {
        Ok(Self {
            candidate: ApprovedBlockCandidate::from_proto(
                proto
                    .candidate
                    .ok_or_else(|| "Missing candidate field".to_string())?,
            )?,
            sig: proto.sig.ok_or_else(|| "Missing sig field".to_string())?,
        })
    }

    pub fn to_proto(self) -> BlockApprovalProto {
        BlockApprovalProto {
            candidate: Some(self.candidate.to_proto()),
            sig: Some(self.sig),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApprovedBlock {
    pub candidate: ApprovedBlockCandidate,
    pub sigs: Vec<Signature>,
}

impl ApprovedBlock {
    pub fn from_proto(proto: ApprovedBlockProto) -> Result<Self, String> {
        Ok(Self {
            candidate: ApprovedBlockCandidate::from_proto(
                proto
                    .candidate
                    .ok_or_else(|| "Missing candidate field".to_string())?,
            )?,
            sigs: proto.sigs,
        })
    }

    pub fn to_proto(self) -> ApprovedBlockProto {
        ApprovedBlockProto {
            candidate: Some(self.candidate.to_proto()),
            sigs: self.sigs,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NoApprovedBlockAvailable {
    pub identifier: String,
    pub node_identifier: String,
}

impl NoApprovedBlockAvailable {
    pub fn from_proto(proto: NoApprovedBlockAvailableProto) -> Self {
        Self {
            identifier: proto.identifier,
            node_identifier: proto.node_identifier,
        }
    }

    pub fn to_proto(self) -> NoApprovedBlockAvailableProto {
        NoApprovedBlockAvailableProto {
            identifier: self.identifier,
            node_identifier: self.node_identifier,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApprovedBlockRequest {
    pub identifier: String,
    pub trim_state: bool,
}

impl ApprovedBlockRequest {
    pub fn from_proto(proto: ApprovedBlockRequestProto) -> Self {
        Self {
            identifier: proto.identifier,
            trim_state: proto.trim_state,
        }
    }

    pub fn to_proto(self) -> ApprovedBlockRequestProto {
        ApprovedBlockRequestProto {
            identifier: self.identifier,
            trim_state: self.trim_state,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockHashMessage {
    pub block_hash: ByteString,
    pub block_creator: ByteString,
}

impl BlockHashMessage {
    pub fn from_proto(proto: BlockHashMessageProto) -> Self {
        Self {
            block_hash: proto.hash,
            block_creator: proto.block_creator,
        }
    }

    pub fn to_proto(self) -> BlockHashMessageProto {
        BlockHashMessageProto {
            hash: self.block_hash,
            block_creator: self.block_creator,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockMessage {
    pub block_hash: ByteString,
    pub header: Header,
    pub body: Body,
    pub justifications: Vec<Justification>,
    pub sender: ByteString,
    pub seq_num: i32,
    pub sig: ByteString,
    pub sig_algorithm: String,
    pub shard_id: String,
    pub extra_bytes: ByteString,
}

impl BlockMessage {
    pub fn from_proto(proto: BlockMessageProto) -> Result<Self, String> {
        Ok(Self {
            block_hash: proto.block_hash,
            header: Header::from_proto(
                proto
                    .header
                    .ok_or_else(|| "Missing header field".to_string())?,
            ),
            body: Body::from_proto(proto.body.ok_or_else(|| "Missing body field".to_string())?)?,
            justifications: proto
                .justifications
                .into_iter()
                .map(|j| Justification::from_proto(j))
                .collect(),
            sender: proto.sender,
            seq_num: proto.seq_num,
            sig: proto.sig,
            sig_algorithm: proto.sig_algorithm,
            shard_id: proto.shard_id,
            extra_bytes: proto.extra_bytes,
        })
    }

    pub fn to_proto(&self) -> BlockMessageProto {
        BlockMessageProto {
            block_hash: self.block_hash.clone(),
            header: Some(self.header.to_proto()),
            body: Some(self.body.to_proto()),
            justifications: self
                .justifications
                .clone()
                .into_iter()
                .map(|j| j.to_proto())
                .collect(),
            sender: self.sender.clone(),
            seq_num: self.seq_num,
            sig: self.sig.clone(),
            sig_algorithm: self.sig_algorithm.clone(),
            shard_id: self.shard_id.clone(),
            extra_bytes: self.extra_bytes.clone(),
        }
    }

    pub fn to_string(self) -> String {
        PrettyPrinter::build_string(CasperMessage::BlockMessage(self), false)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub parents_hash_list: Vec<ByteString>,
    pub timestamp: i64,
    pub version: i64,
    pub extra_bytes: ByteString,
}

impl Header {
    pub fn from_proto(proto: HeaderProto) -> Self {
        Self {
            parents_hash_list: proto.parents_hash_list,
            timestamp: proto.timestamp,
            version: proto.version,
            extra_bytes: proto.extra_bytes,
        }
    }

    pub fn to_proto(&self) -> HeaderProto {
        HeaderProto {
            parents_hash_list: self.parents_hash_list.clone(),
            timestamp: self.timestamp,
            version: self.version,
            extra_bytes: self.extra_bytes.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RejectedDeploy {
    pub sig: ByteString,
}

impl RejectedDeploy {
    pub fn from_proto(proto: RejectedDeployProto) -> Self {
        Self { sig: proto.sig }
    }

    pub fn to_proto(self) -> RejectedDeployProto {
        RejectedDeployProto { sig: self.sig }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub state: F1r3flyState,
    pub deploys: Vec<ProcessedDeploy>,
    pub rejected_deploys: Vec<RejectedDeploy>,
    pub system_deploys: Vec<ProcessedSystemDeploy>,
    pub extra_bytes: ByteString,
}

impl Body {
    pub fn from_proto(proto: BodyProto) -> Result<Self, String> {
        Ok(Self {
            state: F1r3flyState::from_proto(
                proto
                    .state
                    .ok_or_else(|| "Missing state field".to_string())?,
            ),
            deploys: proto
                .deploys
                .into_iter()
                .map(|d| ProcessedDeploy::from_proto(d))
                .collect::<Result<Vec<ProcessedDeploy>, String>>()?,
            rejected_deploys: proto
                .rejected_deploys
                .into_iter()
                .map(|r| RejectedDeploy::from_proto(r))
                .collect(),
            system_deploys: proto
                .system_deploys
                .into_iter()
                .map(|s| ProcessedSystemDeploy::from_proto(s))
                .collect::<Result<Vec<ProcessedSystemDeploy>, String>>()?,
            extra_bytes: proto.extra_bytes,
        })
    }

    pub fn to_proto(&self) -> BodyProto {
        BodyProto {
            state: Some(self.state.to_proto()),
            deploys: self
                .deploys
                .clone()
                .into_iter()
                .map(|d| d.to_proto())
                .collect(),
            rejected_deploys: self
                .rejected_deploys
                .clone()
                .into_iter()
                .map(|r| r.to_proto())
                .collect(),
            system_deploys: self
                .system_deploys
                .clone()
                .into_iter()
                .map(|s| s.to_proto())
                .collect(),
            extra_bytes: self.extra_bytes.clone(),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize, Hash)]
pub struct Justification {
    #[serde(with = "shared::rust::serde_bytes")]
    pub validator: ByteString,
    #[serde(with = "shared::rust::serde_bytes")]
    pub latest_block_hash: ByteString,
}

impl Justification {
    pub fn from_proto(proto: JustificationProto) -> Self {
        Self {
            validator: proto.validator,
            latest_block_hash: proto.latest_block_hash,
        }
    }

    pub fn to_proto(&self) -> JustificationProto {
        JustificationProto {
            validator: self.validator.clone(),
            latest_block_hash: self.latest_block_hash.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct F1r3flyState {
    pub pre_state_hash: ByteString,
    pub post_state_hash: ByteString,
    pub bonds: Vec<Bond>,
    pub block_number: i64,
}

impl F1r3flyState {
    pub fn from_proto(proto: RChainStateProto) -> Self {
        Self {
            pre_state_hash: proto.pre_state_hash,
            post_state_hash: proto.post_state_hash,
            bonds: proto
                .bonds
                .into_iter()
                .map(|b| Bond::from_proto(b))
                .collect(),
            block_number: proto.block_number,
        }
    }

    pub fn to_proto(&self) -> RChainStateProto {
        RChainStateProto {
            pre_state_hash: self.pre_state_hash.clone(),
            post_state_hash: self.post_state_hash.clone(),
            bonds: self
                .bonds
                .clone()
                .into_iter()
                .map(|b| Bond::to_proto(b))
                .collect(),
            block_number: self.block_number,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedDeploy {
    pub deploy: Signed<DeployData>,
    pub cost: PCost,
    pub deploy_log: Vec<Event>,
    pub is_failed: bool,
    pub system_deploy_error: Option<String>,
}

impl ProcessedDeploy {
    pub fn refund_amount(&self) -> i64 {
        (self.deploy.data.phlo_limit - self.cost.cost as i64).max(0) * self.deploy.data.phlo_price
    }

    pub fn empty(deploy: Signed<DeployData>) -> Self {
        Self {
            deploy,
            cost: PCost { cost: 0 },
            deploy_log: Vec::new(),
            is_failed: false,
            system_deploy_error: None,
        }
    }

    pub fn to_deploy_info(self) -> DeployInfo {
        DeployInfo {
            deployer: PrettyPrinter::build_string_no_limit(&self.deploy.pk.bytes),
            term: self.deploy.data.term.clone(),
            timestamp: self.deploy.data.time_stamp,
            sig: PrettyPrinter::build_string_no_limit(&self.deploy.sig),
            sig_algorithm: self.deploy.sig_algorithm.name(),
            phlo_price: self.deploy.data.phlo_price,
            phlo_limit: self.deploy.data.phlo_limit,
            valid_after_block_number: self.deploy.data.valid_after_block_number,
            cost: self.cost.cost,
            errored: self.is_failed,
            system_deploy_error: self.system_deploy_error.unwrap_or_default(),
        }
    }

    pub fn from_proto(proto: ProcessedDeployProto) -> Result<Self, String> {
        Ok(Self {
            deploy: DeployData::from_proto(
                proto
                    .deploy
                    .ok_or_else(|| "Missing deploy field".to_string())?,
            )?,
            cost: proto.cost.ok_or_else(|| "Missing cost field".to_string())?,
            deploy_log: proto
                .deploy_log
                .into_iter()
                .map(|e| Event::from_proto(e))
                .collect::<Result<Vec<Event>, String>>()?,
            is_failed: proto.errored,
            system_deploy_error: {
                if proto.system_deploy_error.is_empty() {
                    None
                } else {
                    Some(proto.system_deploy_error)
                }
            },
        })
    }

    pub fn to_proto(self) -> ProcessedDeployProto {
        ProcessedDeployProto {
            deploy: Some(DeployData::to_proto(self.deploy)),
            cost: Some(self.cost),
            deploy_log: self.deploy_log.into_iter().map(|e| e.to_proto()).collect(),
            errored: self.is_failed,
            system_deploy_error: self.system_deploy_error.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemDeployData {
    Slash {
        invalid_block_hash: ByteString,
        issuer_public_key: PublicKey,
    },
    CloseBlockSystemDeployData,
    Empty,
}

impl SystemDeployData {
    pub fn create_slash(invalid_block_hash: ByteString, issuer_public_key: PublicKey) -> Self {
        Self::Slash {
            invalid_block_hash,
            issuer_public_key,
        }
    }

    pub fn create_close() -> Self {
        Self::CloseBlockSystemDeployData
    }

    pub fn from_proto(proto: SystemDeployDataProto) -> Result<Self, String> {
        match proto
            .system_deploy
            .ok_or_else(|| "Missing system deploy field".to_string())?
        {
            system_deploy_data_proto::SystemDeploy::SlashSystemDeploy(
                slash_system_deploy_data_proto,
            ) => Ok(Self::Slash {
                invalid_block_hash: slash_system_deploy_data_proto.invalid_block_hash,
                issuer_public_key: PublicKey::from_bytes(
                    &slash_system_deploy_data_proto.issuer_public_key,
                ),
            }),
            system_deploy_data_proto::SystemDeploy::CloseBlockSystemDeploy(_) => {
                Ok(Self::CloseBlockSystemDeployData)
            }
        }
    }

    pub fn to_proto(sdd: SystemDeployData) -> SystemDeployDataProto {
        match sdd {
            Self::Slash {
                invalid_block_hash,
                issuer_public_key,
            } => SystemDeployDataProto {
                system_deploy: Some(SystemDeploy::SlashSystemDeploy(
                    SlashSystemDeployDataProto {
                        invalid_block_hash,
                        issuer_public_key: issuer_public_key.bytes.into(),
                    },
                )),
            },
            Self::CloseBlockSystemDeployData => SystemDeployDataProto {
                system_deploy: Some(SystemDeploy::CloseBlockSystemDeploy(
                    CloseBlockSystemDeployDataProto {},
                )),
            },
            Self::Empty => SystemDeployDataProto {
                system_deploy: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessedSystemDeploy {
    Succeeded {
        event_list: Vec<Event>,
        system_deploy: SystemDeployData,
    },
    Failed {
        event_list: Vec<Event>,
        error_msg: String,
    },
}

impl ProcessedSystemDeploy {
    pub fn failed(self) -> bool {
        matches!(self, ProcessedSystemDeploy::Failed { .. })
    }

    pub fn fold<A, F, G>(self, if_succeeded: F, if_failed: G) -> A
    where
        F: Fn(Vec<Event>) -> A,
        G: Fn(Vec<Event>, String) -> A,
    {
        match self {
            ProcessedSystemDeploy::Succeeded { event_list, .. } => if_succeeded(event_list),
            ProcessedSystemDeploy::Failed {
                event_list,
                error_msg,
            } => if_failed(event_list, error_msg),
        }
    }

    pub fn from_proto(psd: ProcessedSystemDeployProto) -> Result<Self, String> {
        let deploy_log: Result<Vec<Event>, String> =
            psd.deploy_log.into_iter().map(Event::from_proto).collect();

        match deploy_log {
            Ok(deploy_log) => {
                if psd.error_msg.is_empty() {
                    Ok(ProcessedSystemDeploy::Succeeded {
                        event_list: deploy_log,
                        system_deploy: SystemDeployData::from_proto(
                            psd.system_deploy
                                .ok_or_else(|| "Missing system deploy field".to_string())?,
                        )?,
                    })
                } else {
                    Ok(ProcessedSystemDeploy::Failed {
                        event_list: deploy_log,
                        error_msg: psd.error_msg,
                    })
                }
            }
            Err(err) => Err(err),
        }
    }

    pub fn to_proto(self) -> ProcessedSystemDeployProto {
        match self {
            ProcessedSystemDeploy::Succeeded {
                event_list,
                system_deploy,
            } => ProcessedSystemDeployProto {
                system_deploy: Some(SystemDeployData::to_proto(system_deploy)),
                deploy_log: event_list
                    .into_iter()
                    .map(|arg0: Event| Event::to_proto(&arg0))
                    .collect(),
                error_msg: "".to_string(),
            },
            ProcessedSystemDeploy::Failed {
                event_list,
                error_msg,
            } => ProcessedSystemDeployProto {
                system_deploy: None,
                deploy_log: event_list
                    .into_iter()
                    .map(|arg0: Event| Event::to_proto(&arg0))
                    .collect(),
                error_msg,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, Eq, Hash)]
pub struct DeployData {
    pub term: String,
    pub time_stamp: i64,
    pub phlo_price: i64,
    pub phlo_limit: i64,
    pub valid_after_block_number: i64,
    pub shard_id: String,
    pub language: String,
}

impl DeployData {
    pub fn total_phlo_charge(&self) -> i64 {
        self.phlo_limit * self.phlo_price
    }

    pub fn encode(a: DeployData) -> ByteVector {
        DeployData::_to_proto(a).encode_to_vec()
    }

    pub fn decode(a: ByteVector) -> Result<DeployData, String> {
        let proto = DeployDataProto::decode(&a[..])
            .map_err(|e| format!("Failed to decode DeployData: {}", e))?;
        Ok(DeployData::_from_proto(proto))
    }

    fn _from_proto(proto: DeployDataProto) -> Self {
        Self {
            term: proto.term,
            time_stamp: proto.timestamp,
            phlo_price: proto.phlo_price,
            phlo_limit: proto.phlo_limit,
            valid_after_block_number: proto.valid_after_block_number,
            shard_id: proto.shard_id,
            language: proto.language,
        }
    }

    pub fn from_proto(proto: DeployDataProto) -> Result<Signed<DeployData>, String> {
        let algorithm = SignaturesAlgFactory::apply(&proto.sig_algorithm)
            .ok_or_else(|| format!("Unknown signature algorithm: {}", proto.sig_algorithm))?;

        let sig = proto.sig.clone();
        let pk = PublicKey::from_bytes(&proto.deployer);
        let signed = Signed::from_signed_data(DeployData::_from_proto(proto), pk, sig, algorithm)?;

        match signed {
            Some(signed) => Ok(signed),
            None => Err("Invalid signature".to_string()),
        }
    }

    fn _to_proto(dd: DeployData) -> DeployDataProto {
        DeployDataProto {
            term: dd.term,
            timestamp: dd.time_stamp,
            phlo_price: dd.phlo_price,
            phlo_limit: dd.phlo_limit,
            valid_after_block_number: dd.valid_after_block_number,
            shard_id: dd.shard_id,
            language: dd.language,
            ..Default::default()
        }
    }

    pub fn to_proto(dd: Signed<DeployData>) -> DeployDataProto {
        DeployDataProto {
            term: dd.data.term.clone(),
            timestamp: dd.data.time_stamp,
            phlo_price: dd.data.phlo_price,
            phlo_limit: dd.data.phlo_limit,
            valid_after_block_number: dd.data.valid_after_block_number,
            shard_id: dd.data.shard_id.clone(),
            deployer: dd.pk.bytes.clone().into(),
            sig: dd.sig.clone().into(),
            sig_algorithm: dd.sig_algorithm.name(),
            language: dd.data.language.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Peek {
    pub channel_index: i32,
}

impl Peek {
    pub fn from_proto(proto: PeekProto) -> Self {
        Self {
            channel_index: proto.channel_index,
        }
    }

    pub fn to_proto(self) -> PeekProto {
        PeekProto {
            channel_index: self.channel_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Produce(ProduceEvent),
    Consume(ConsumeEvent),
    Comm(CommEvent),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProduceEvent {
    pub channels_hash: ByteString,
    pub hash: ByteString,
    pub persistent: bool,
    pub times_repeated: i32,
    pub is_deterministic: bool,
    pub output_value: Vec<ByteString>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConsumeEvent {
    pub channels_hashes: Vec<ByteString>,
    pub hash: ByteString,
    pub persistent: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommEvent {
    pub consume: ConsumeEvent,
    pub produces: Vec<ProduceEvent>,
    pub peeks: Vec<Peek>,
}

impl Event {
    pub fn from_proto(proto: EventProto) -> Result<Event, String> {
        match proto.event_instance {
            Some(event_proto::EventInstance::Produce(pe)) => {
                Ok(Event::Produce(ProduceEvent::from_proto(pe)))
            }
            Some(event_proto::EventInstance::Consume(ce)) => {
                Ok(Event::Consume(ConsumeEvent::from_proto(ce)))
            }
            Some(event_proto::EventInstance::Comm(CommEventProto {
                consume,
                produces,
                peeks,
            })) => Ok(Event::Comm(CommEvent {
                consume: ConsumeEvent::from_proto(
                    consume.ok_or_else(|| "Missing consume field".to_string())?,
                ),
                produces: produces.into_iter().map(ProduceEvent::from_proto).collect(),
                peeks: peeks.into_iter().map(Peek::from_proto).collect(),
            })),

            _ => Err("Received malformed Event: None".to_string()),
        }
    }

    pub fn to_proto(&self) -> EventProto {
        match self {
            Event::Produce(pe) => EventProto {
                event_instance: Some(event_proto::EventInstance::Produce(pe.clone().to_proto())),
            },
            Event::Consume(ce) => EventProto {
                event_instance: Some(event_proto::EventInstance::Consume(ce.clone().to_proto())),
            },
            Event::Comm(cme) => EventProto {
                event_instance: Some(event_proto::EventInstance::Comm(cme.clone().to_proto())),
            },
        }
    }
}

impl ProduceEvent {
    pub fn to_proto(self) -> ProduceEventProto {
        ProduceEventProto {
            channels_hash: self.channels_hash,
            hash: self.hash,
            persistent: self.persistent,
            times_repeated: self.times_repeated,
            is_deterministic: self.is_deterministic,
            output_value: self.output_value,
        }
    }

    pub fn from_proto(proto: ProduceEventProto) -> Self {
        ProduceEvent {
            channels_hash: proto.channels_hash,
            hash: proto.hash,
            persistent: proto.persistent,
            times_repeated: proto.times_repeated,
            is_deterministic: proto.is_deterministic,
            output_value: proto.output_value,
        }
    }
}

impl ConsumeEvent {
    pub fn to_proto(self) -> ConsumeEventProto {
        ConsumeEventProto {
            channels_hashes: self.channels_hashes,
            hash: self.hash,
            persistent: self.persistent,
        }
    }

    pub fn from_proto(proto: ConsumeEventProto) -> Self {
        ConsumeEvent {
            channels_hashes: proto.channels_hashes,
            hash: proto.hash,
            persistent: proto.persistent,
        }
    }
}

impl CommEvent {
    pub fn to_proto(self) -> CommEventProto {
        CommEventProto {
            consume: Some(self.consume.to_proto()),
            produces: self.produces.into_iter().map(|pe| pe.to_proto()).collect(),
            peeks: self.peeks.into_iter().map(|pk| pk.to_proto()).collect(),
        }
    }

    pub fn from_proto(
        consume: ConsumeEventProto,
        produces: Vec<ProduceEventProto>,
        peeks: Vec<PeekProto>,
    ) -> Self {
        CommEvent {
            consume: ConsumeEvent::from_proto(consume),
            produces: produces.into_iter().map(ProduceEvent::from_proto).collect(),
            peeks: peeks.into_iter().map(Peek::from_proto).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bond {
    pub validator: ByteString,
    pub stake: i64,
}

impl Bond {
    pub fn from_proto(proto: BondProto) -> Self {
        Self {
            validator: proto.validator,
            stake: proto.stake,
        }
    }

    pub fn to_proto(self) -> BondProto {
        BondProto {
            validator: self.validator,
            stake: self.stake,
        }
    }
}

// Last finalized state

pub struct StoreNodeKey {
    pub hash: Blake2b256Hash,
    pub index: Option<Byte>,
}

impl StoreNodeKey {
    // Encoding of non-existent index for store node (Skip or Leaf node)
    const NONE_INDEX: i32 = 0x100;

    pub fn from_proto(proto: StoreNodeKeyProto) -> (Blake2b256Hash, Option<Byte>) {
        // Key hash
        let hash_bytes = Blake2b256Hash::from_bytes(proto.hash.to_vec());

        // Relative branch index / max 8-bit
        let idx = if proto.index == Self::NONE_INDEX {
            None
        } else {
            Some(proto.index as u8)
        };

        (hash_bytes, idx)
    }

    pub fn to_proto(s: &(Blake2b256Hash, Option<Byte>)) -> StoreNodeKeyProto {
        StoreNodeKeyProto {
            hash: s.0.bytes().into(),
            index: s.1.map(|b| b).unwrap_or(Self::NONE_INDEX as u8) as i32,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoreItemsMessageRequest {
    pub start_path: Vec<(Blake2b256Hash, Option<Byte>)>,
    pub skip: i32,
    pub take: i32,
}

impl StoreItemsMessageRequest {
    pub fn from_proto(proto: StoreItemsMessageRequestProto) -> Self {
        Self {
            start_path: proto
                .start_path
                .into_iter()
                .map(StoreNodeKey::from_proto)
                .collect(),
            skip: proto.skip,
            take: proto.take,
        }
    }

    pub fn to_proto(self) -> StoreItemsMessageRequestProto {
        StoreItemsMessageRequestProto {
            start_path: self.start_path.iter().map(StoreNodeKey::to_proto).collect(),
            skip: self.skip,
            take: self.take,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoreItemsMessage {
    pub start_path: Vec<(Blake2b256Hash, Option<Byte>)>,
    pub last_path: Vec<(Blake2b256Hash, Option<Byte>)>,
    pub history_items: Vec<(Blake2b256Hash, ByteString)>,
    pub data_items: Vec<(Blake2b256Hash, ByteString)>,
}

impl StoreItemsMessage {
    pub fn pretty(self) -> String {
        let start: String = self
            .start_path
            .iter()
            .map(RSpaceExporterInstance::path_pretty)
            .collect();

        let last: String = self
            .last_path
            .iter()
            .map(RSpaceExporterInstance::path_pretty)
            .collect();

        let history_size = self.history_items.len();
        let data_size = self.data_items.len();

        format!(
            "StoreItemsMessage(history: {:?}, data: {:?}, start: {:?}, last: {:?})",
            history_size, data_size, start, last
        )
    }

    pub fn from_proto(proto: StoreItemsMessageProto) -> Self {
        Self {
            start_path: proto
                .start_path
                .into_iter()
                .map(StoreNodeKey::from_proto)
                .collect(),
            last_path: proto
                .last_path
                .into_iter()
                .map(StoreNodeKey::from_proto)
                .collect(),
            history_items: proto
                .history_items
                .into_iter()
                .map(|store_item_proto| {
                    (
                        Blake2b256Hash::from_bytes(store_item_proto.key.to_vec()),
                        store_item_proto.value,
                    )
                })
                .collect(),
            data_items: proto
                .data_items
                .into_iter()
                .map(|store_item_proto| {
                    (
                        Blake2b256Hash::from_bytes(store_item_proto.key.to_vec()),
                        store_item_proto.value,
                    )
                })
                .collect(),
        }
    }

    pub fn to_proto(self) -> StoreItemsMessageProto {
        StoreItemsMessageProto {
            start_path: self.start_path.iter().map(StoreNodeKey::to_proto).collect(),
            last_path: self.last_path.iter().map(StoreNodeKey::to_proto).collect(),
            history_items: self
                .history_items
                .into_iter()
                .map(|(key, value)| StoreItemProto {
                    key: key.bytes().into(),
                    value,
                })
                .collect(),
            data_items: self
                .data_items
                .into_iter()
                .map(|(key, value)| StoreItemProto {
                    key: key.bytes().into(),
                    value,
                })
                .collect(),
        }
    }
}
