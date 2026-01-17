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
pub mod heartbeat_signal;
pub mod helper;
pub mod last_finalized_height_constraint_checker;
pub mod merging;
pub mod metrics_constants;
pub mod multi_parent_casper_impl;
pub mod protocol;
pub mod report_store;
pub mod reporting_casper;
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
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::rust::{
    block_status::{BlockError, ValidBlock},
    blocks::proposer::proposer::ProposerResult,
    casper::MultiParentCasper,
    errors::CasperError,
};

pub type TopoSort = Vec<Vec<BlockHash>>;

pub type BlockProcessing<A> = Either<BlockError, A>;

pub type ValidBlockProcessing = BlockProcessing<ValidBlock>;

// Async function that takes Arc<dyn MultiParentCasper> by value and boolean, returns Future of ProposerResult
pub type ProposeFunction = dyn Fn(
        Arc<dyn MultiParentCasper + Send + Sync>,
        bool,
    ) -> Pin<Box<dyn Future<Output = Result<ProposerResult, CasperError>> + Send>>
    + Send
    + Sync;
