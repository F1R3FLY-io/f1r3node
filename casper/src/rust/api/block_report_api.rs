// See casper/src/main/scala/coop/rchain/casper/api/BlockReportAPI.scala

use models::{casper::BlockEventInfo, rust::block_hash::BlockHash};

// TODO: provide the related implementation. Current stub was added in scope of porting the node/src/rust/web/transaction.rs,
// where this BlockReportAPI struct is used.
pub struct BlockReportAPI {}

impl BlockReportAPI {
    pub async fn block_report(
        &self,
        _block_message: BlockHash,
        _force_replay: bool,
    ) -> eyre::Result<BlockEventInfo> {
        todo!()
    }
}
