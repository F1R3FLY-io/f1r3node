// See casper/src/main/scala/coop/rchain/casper/state/instances/ProposerState.scala

use crate::rust::blocks::proposer::propose_result::ProposeResult;
use models::rust::casper::protocol::casper_message::BlockMessage;
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct ProposerState {
    pub latest_propose_result: Option<(ProposeResult, Option<BlockMessage>)>,
    pub curr_propose_result: Option<oneshot::Receiver<(ProposeResult, Option<BlockMessage>)>>,
}

impl Default for ProposerState {
    fn default() -> Self {
        Self {
            latest_propose_result: None,
            curr_propose_result: None,
        }
    }
}
