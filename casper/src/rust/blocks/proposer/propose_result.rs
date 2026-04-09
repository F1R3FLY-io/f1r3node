// See casper/src/main/scala/coop/rchain/casper/blocks/proposer/ProposeResult.scala

use std::fmt;
use uuid::Uuid;

use models::rust::casper::protocol::casper_message::BlockMessage;
use prost::bytes::Bytes;

use crate::rust::block_status::ValidBlock;

/// Propose ID type
pub type ProposeID = Uuid;

/// Result of a block proposal attempt
#[derive(Debug, Clone)]
pub struct ProposeResult {
    pub propose_status: ProposeStatus,
}

/// Status of a block proposal
#[derive(Debug, Clone)]
pub enum ProposeStatus {
    Success(ProposeSuccess),
    Failure(ProposeFailure),
}

/// Successful proposal
#[derive(Debug, Clone)]
pub struct ProposeSuccess {
    pub result: ValidBlock,
}

/// Failed proposal
#[derive(Debug, Clone)]
pub enum ProposeFailure {
    NoNewDeploys,
    InternalDeployError,
    BugError,
    CheckConstraintsFailure(CheckProposeConstraintsFailure),
}

/// Check constraints result
#[derive(Debug, Clone)]
pub enum CheckProposeConstraintsResult {
    Success,
    Failure(CheckProposeConstraintsFailure),
}

/// Constraints check failure
#[derive(Debug, Clone)]
pub enum CheckProposeConstraintsFailure {
    NotBonded,
    NotEnoughNewBlocks,
    TooFarAheadOfLastFinalized,
}

/// Block creator result
#[derive(Debug, Clone)]
pub enum BlockCreatorResult {
    NoNewDeploys,
    /// The created block together with the pre- and post-state hashes that were computed
    /// during `compute_deploys_checkpoint`. Carrying these hashes avoids re-running the
    /// expensive checkpoint replay during self-validation.
    Created(BlockMessage, Bytes, Bytes),
}

impl CheckProposeConstraintsResult {
    pub fn success() -> Self {
        CheckProposeConstraintsResult::Success
    }

    pub fn not_bonded() -> Self {
        CheckProposeConstraintsResult::Failure(CheckProposeConstraintsFailure::NotBonded)
    }

    pub fn not_enough_new_block() -> Self {
        CheckProposeConstraintsResult::Failure(CheckProposeConstraintsFailure::NotEnoughNewBlocks)
    }

    pub fn too_far_ahead_of_last_finalized() -> Self {
        CheckProposeConstraintsResult::Failure(
            CheckProposeConstraintsFailure::TooFarAheadOfLastFinalized,
        )
    }
}

impl ProposeResult {
    pub fn no_new_deploys() -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(ProposeFailure::NoNewDeploys),
        }
    }

    pub fn internal_deploy_error() -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(ProposeFailure::InternalDeployError),
        }
    }

    pub fn not_bonded() -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                CheckProposeConstraintsFailure::NotBonded,
            )),
        }
    }

    pub fn not_enough_blocks() -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                CheckProposeConstraintsFailure::NotEnoughNewBlocks,
            )),
        }
    }

    pub fn too_far_ahead_of_last_finalized() -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(ProposeFailure::CheckConstraintsFailure(
                CheckProposeConstraintsFailure::TooFarAheadOfLastFinalized,
            )),
        }
    }

    pub fn success(status: ValidBlock) -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Success(ProposeSuccess { result: status }),
        }
    }

    pub fn failure(status: ProposeFailure) -> Self {
        ProposeResult {
            propose_status: ProposeStatus::Failure(status),
        }
    }

    pub fn is_no_new_deploys(&self) -> bool {
        matches!(
            self.propose_status,
            ProposeStatus::Failure(ProposeFailure::NoNewDeploys)
        )
    }
}

impl BlockCreatorResult {
    pub fn no_new_deploys() -> Self {
        BlockCreatorResult::NoNewDeploys
    }

    pub fn created(b: BlockMessage, pre_state_hash: Bytes, post_state_hash: Bytes) -> Self {
        BlockCreatorResult::Created(b, pre_state_hash, post_state_hash)
    }
}

impl fmt::Display for ProposeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProposeStatus::Success(r) => write!(f, "Propose succeed: {:?}", r.result),
            ProposeStatus::Failure(failure) => match failure {
                ProposeFailure::NoNewDeploys => write!(f, "Proposal failed: NoNewDeploys. No unprocessed deploys in pool. If you just deployed, the deploy may have already been included by the auto-proposer."),
                ProposeFailure::InternalDeployError => {
                    write!(f, "Proposal failed: internal deploy error")
                }
                ProposeFailure::BugError => write!(f, "Proposal failed: BugError"),
                ProposeFailure::CheckConstraintsFailure(check_failure) => match check_failure {
                    CheckProposeConstraintsFailure::NotBonded => {
                        write!(f, "Proposal failed: ReadOnlyMode")
                    }
                    CheckProposeConstraintsFailure::NotEnoughNewBlocks => {
                        write!(
                            f,
                            "Proposal failed: Must wait for more blocks from other validators"
                        )
                    }
                    CheckProposeConstraintsFailure::TooFarAheadOfLastFinalized => {
                        write!(
                            f,
                            "Proposal failed: too far ahead of the last finalized block"
                        )
                    }
                },
            },
        }
    }
}
