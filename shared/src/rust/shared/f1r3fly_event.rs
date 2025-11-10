// See shared/src/main/scala/coop/rchain/shared/RChainEvent.scala

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum F1r3flyEvent {
    BlockCreated(BlockCreated),
    BlockAdded(BlockAdded),
    BlockFinalised(BlockFinalised),
    SentUnapprovedBlock(SentUnapprovedBlockData),
    SentApprovedBlock(SentApprovedBlockData),
    BlockApprovalReceived(BlockApprovalReceived),
    ApprovedBlockReceived(ApprovedBlockReceived),
    EnteredRunningState(EnteredRunningState),
    NodeStarted(NodeStarted),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BlockCreated {
    pub block_hash: String,
    pub parent_hashes: Vec<String>,
    pub justification_hashes: Vec<(String, String)>,
    pub deploy_ids: Vec<String>,
    pub creator: String,
    pub seq_number: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BlockAdded {
    pub block_hash: String,
    pub parent_hashes: Vec<String>,
    pub justification_hashes: Vec<(String, String)>,
    pub deploy_ids: Vec<String>,
    pub creator: String,
    pub seq_number: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BlockFinalised {
    pub block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BlockApprovalReceived {
    pub block_hash: String,
    pub sender: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ApprovedBlockReceived {
    pub block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct EnteredRunningState {
    pub block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SentUnapprovedBlockData {
    pub block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SentApprovedBlockData {
    pub block_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct NodeStarted {
    pub address: String,
}

impl F1r3flyEvent {
    pub fn block_created(
        block_hash: String,
        parent_hashes: Vec<String>,
        justification_hashes: Vec<(String, String)>,
        deploy_ids: Vec<String>,
        creator: String,
        seq_number: i32,
    ) -> Self {
        Self::BlockCreated(BlockCreated {
            block_hash,
            parent_hashes,
            justification_hashes,
            deploy_ids,
            creator,
            seq_number,
        })
    }

    pub fn block_added(
        block_hash: String,
        parent_hashes: Vec<String>,
        justification_hashes: Vec<(String, String)>,
        deploy_ids: Vec<String>,
        creator: String,
        seq_number: i32,
    ) -> Self {
        Self::BlockAdded(BlockAdded {
            block_hash,
            parent_hashes,
            justification_hashes,
            deploy_ids,
            creator,
            seq_number,
        })
    }

    pub fn block_finalised(block_hash: String) -> Self {
        Self::BlockFinalised(BlockFinalised { block_hash })
    }

    pub fn approved_block_received(block_hash: String) -> Self {
        Self::ApprovedBlockReceived(ApprovedBlockReceived { block_hash })
    }

    pub fn entered_running_state(block_hash: String) -> Self {
        Self::EnteredRunningState(EnteredRunningState { block_hash })
    }

    pub fn sent_unapproved_block(block_hash: String) -> Self {
        Self::SentUnapprovedBlock(SentUnapprovedBlockData { block_hash })
    }

    pub fn sent_approved_block(block_hash: String) -> Self {
        Self::SentApprovedBlock(SentApprovedBlockData { block_hash })
    }

    pub fn block_approval_received(block_hash: String, sender: String) -> Self {
        Self::BlockApprovalReceived(BlockApprovalReceived { block_hash, sender })
    }

    pub fn node_started(address: String) -> Self {
        Self::NodeStarted(NodeStarted { address })
    }
}
