// See casper/src/main/scala/coop/rchain/casper/api/BlockReportAPI.scala

use std::sync::Arc;

use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use models::{casper::BlockEventInfo, rust::block_hash::BlockHash};

use crate::rust::{
    engine::engine_cell::EngineCell,
    report_store::ReportStore,
    reporting_casper::ReportingCasper,
    safety_oracle::CliqueOracleImpl,
};

// TODO: provide the related implementation. Current stub was added in scope of porting the node/src/rust/web/transaction.rs,
// where this BlockReportAPI struct is used.
#[derive(Clone)]
pub struct BlockReportAPI {
    #[allow(dead_code)]
    reporting_casper: Arc<dyn ReportingCasper>,
    #[allow(dead_code)]
    report_store: ReportStore,
    #[allow(dead_code)]
    engine_cell: EngineCell,
    #[allow(dead_code)]
    block_store: KeyValueBlockStore,
    #[allow(dead_code)]
    oracle: CliqueOracleImpl,
}

impl BlockReportAPI {
    /// Create a new BlockReportAPI
    /// 
    /// Scala: BlockReportAPI[F](reportingCasper, reportStore)(implicit engineCell, blockStore, oracle)
    pub fn new(
        reporting_casper: Arc<dyn ReportingCasper>,
        report_store: ReportStore,
        engine_cell: EngineCell,
        block_store: KeyValueBlockStore,
        oracle: CliqueOracleImpl,
    ) -> Self {
        Self {
            reporting_casper,
            report_store,
            engine_cell,
            block_store,
            oracle,
        }
    }

    pub async fn block_report(
        &self,
        _block_message: BlockHash,
        _force_replay: bool,
    ) -> eyre::Result<BlockEventInfo> {
        todo!()
    }
}
