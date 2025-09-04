// See casper/src/main/scala/coop/rchain/casper/BlockStatus.scala

use super::errors::CasperError;
use shared::rust::store::key_value_store::KvStoreError;

/// Represents the status of a block in the system
#[derive(Debug, Clone)]
pub enum BlockStatus {
    Valid(ValidBlock),
    Error(BlockError),
}

/// Represents a valid block
#[derive(Debug, Clone, PartialEq)]
pub enum ValidBlock {
    Valid,
}

/// Represents an error with a block
#[derive(Debug, Clone, PartialEq)]
pub enum BlockError {
    Processed,
    CasperIsBusy,
    MissingBlocks,
    BlockException(CasperError),
    Invalid(InvalidBlock),
}

/// Represents an invalid block
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InvalidBlock {
    // AdmissibleEquivocation are blocks that would create an equivocation but are
    // pulled in through a justification of another block
    AdmissibleEquivocation,
    // TODO: Make IgnorableEquivocation slashable again and remember to add an entry to the equivocation record.
    // For now we won't eagerly slash equivocations that we can just ignore,
    // as we aren't forced to add it to our view as a dependency.
    // TODO: The above will become a DOS vector if we don't fix.
    IgnorableEquivocation,

    InvalidFormat,
    InvalidSignature,
    InvalidSender,
    InvalidVersion,
    InvalidTimestamp,

    DeployNotSigned,
    InvalidBlockNumber,
    InvalidRepeatDeploy,
    InvalidParents,
    InvalidFollows,
    InvalidSequenceNumber,
    InvalidShardId,
    JustificationRegression,
    NeglectedInvalidBlock,
    NeglectedEquivocation,
    InvalidTransaction,
    InvalidBondsCache,
    InvalidBlockHash,
    InvalidRejectedDeploy,
    ContainsExpiredDeploy,
    ContainsFutureDeploy,
    NotOfInterest,
    LowDeployCost,
}

impl BlockStatus {
    pub fn valid() -> ValidBlock {
        ValidBlock::Valid
    }

    pub fn processed() -> BlockError {
        BlockError::Processed
    }

    pub fn casper_is_busy() -> BlockError {
        BlockError::CasperIsBusy
    }

    pub fn exception(ex: CasperError) -> BlockError {
        BlockError::BlockException(ex)
    }

    pub fn missing_blocks() -> BlockError {
        BlockError::MissingBlocks
    }

    pub fn admissible_equivocation() -> BlockError {
        BlockError::Invalid(InvalidBlock::AdmissibleEquivocation)
    }

    pub fn ignorable_equivocation() -> BlockError {
        BlockError::Invalid(InvalidBlock::IgnorableEquivocation)
    }

    pub fn invalid_format() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidFormat)
    }

    pub fn invalid_signature() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidSignature)
    }

    pub fn invalid_sender() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidSender)
    }

    pub fn invalid_version() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidVersion)
    }

    pub fn invalid_timestamp() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidTimestamp)
    }

    pub fn deploy_not_signed() -> BlockError {
        BlockError::Invalid(InvalidBlock::DeployNotSigned)
    }

    pub fn invalid_block_number() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidBlockNumber)
    }

    pub fn invalid_repeat_deploy() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidRepeatDeploy)
    }

    pub fn invalid_parents() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidParents)
    }

    pub fn invalid_follows() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidFollows)
    }

    pub fn invalid_sequence_number() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidSequenceNumber)
    }

    pub fn invalid_shard_id() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidShardId)
    }

    pub fn justification_regression() -> BlockError {
        BlockError::Invalid(InvalidBlock::JustificationRegression)
    }

    pub fn neglected_invalid_block() -> BlockError {
        BlockError::Invalid(InvalidBlock::NeglectedInvalidBlock)
    }

    pub fn neglected_equivocation() -> BlockError {
        BlockError::Invalid(InvalidBlock::NeglectedEquivocation)
    }

    pub fn invalid_transaction() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidTransaction)
    }

    pub fn invalid_bonds_cache() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidBondsCache)
    }

    pub fn invalid_block_hash() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidBlockHash)
    }

    pub fn invalid_rejected_deploy() -> BlockError {
        BlockError::Invalid(InvalidBlock::InvalidRejectedDeploy)
    }

    pub fn contains_expired_deploy() -> BlockError {
        BlockError::Invalid(InvalidBlock::ContainsExpiredDeploy)
    }

    pub fn contains_future_deploy() -> BlockError {
        BlockError::Invalid(InvalidBlock::ContainsFutureDeploy)
    }

    pub fn not_of_interest() -> BlockError {
        BlockError::Invalid(InvalidBlock::NotOfInterest)
    }

    pub fn low_deploy_cost() -> BlockError {
        BlockError::Invalid(InvalidBlock::LowDeployCost)
    }

    pub fn is_in_dag(&self) -> bool {
        match self {
            BlockStatus::Valid(_) => true,
            BlockStatus::Error(BlockError::Invalid(_)) => true,
            _ => false,
        }
    }
}

impl InvalidBlock {
    pub fn is_slashable(&self) -> bool {
        match self {
            InvalidBlock::AdmissibleEquivocation
            | InvalidBlock::DeployNotSigned
            | InvalidBlock::InvalidBlockNumber
            | InvalidBlock::InvalidRepeatDeploy
            | InvalidBlock::InvalidParents
            | InvalidBlock::InvalidFollows
            | InvalidBlock::InvalidSequenceNumber
            | InvalidBlock::InvalidShardId
            | InvalidBlock::JustificationRegression
            | InvalidBlock::NeglectedInvalidBlock
            | InvalidBlock::NeglectedEquivocation
            | InvalidBlock::InvalidTransaction
            | InvalidBlock::InvalidBondsCache
            | InvalidBlock::InvalidBlockHash
            | InvalidBlock::ContainsExpiredDeploy
            | InvalidBlock::ContainsFutureDeploy => true,
            _ => false,
        }
    }
}

impl From<KvStoreError> for BlockError {
    fn from(error: KvStoreError) -> Self {
        BlockError::BlockException(CasperError::from(error))
    }
}
