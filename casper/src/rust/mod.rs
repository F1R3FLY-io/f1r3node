pub mod api;
pub mod block_status;
pub mod blocks;
pub mod casper;
pub mod casper_conf;
pub mod engine;
pub mod equivocation_detector;
pub mod errors;
pub mod estimator;
pub mod finality;
pub mod genesis;
pub mod helper;
pub mod last_finalized_height_constraint_checker;
pub mod merging;
pub mod multi_parent_casper_impl;
pub mod protocol;
pub mod reporting_proto_transformer;
pub mod rholang;
pub mod safety;
pub mod safety_oracle;
pub mod state;
pub mod storage;
pub mod synchrony_constraint_checker;
pub mod util;
pub mod validate;
pub mod validator_identity;

// See casper/src/main/scala/coop/rchain/casper/package.scala

use models::rust::block_hash::BlockHash;
use rspace_plus_plus::rspace::history::Either;

use crate::rust::{
    block_status::{BlockError, ValidBlock},
    blocks::proposer::proposer::ProposerResult,
    casper::{Casper, MultiParentCasper},
    errors::CasperError,
};

pub type TopoSort = Vec<Vec<BlockHash>>;

pub type BlockProcessing<A> = Either<BlockError, A>;

pub type ValidBlockProcessing = BlockProcessing<ValidBlock>;

pub type ProposeFunction =
    dyn Fn(&dyn MultiParentCasper, bool) -> Result<ProposerResult, CasperError> + Send + Sync;
